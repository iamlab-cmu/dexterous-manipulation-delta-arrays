import numpy as np
from copy import deepcopy
import itertools
import wandb
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils.MADPTD3.dit_core as core
import utils.multi_agent_replay_buffer as MARB
import utils.loss_utils as loss_utils

class MADPTD3:
    def __init__(self, env_dict, hp_dict, logger):
        self.logger = logger
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.infer_every = hp_dict['infer_every']
        self.gamma = hp_dict['gamma']
        self.alpha = hp_dict['alpha']
        self.max_avg_rew = 0
        self.batch_size = hp_dict['batch_size']
        self.denoising_params = hp_dict['denoising_params']
        self.diff_timesteps = self.denoising_params['n_diff_steps']
        self.policy_delay = hp_dict['policy_delay']
        self.k_thresh = hp_dict['k_thresh']
        self.ppo_clip = hp_dict['ppo_clip']
        
        self.w_k = hp_dict['w_k']
        
        if self.hp_dict['data_type'] == "image":
            self.ma_replay_buffer = MARB.MultiAgentImageReplayBuffer(act_dim=self.act_dim, size=hp_dict['rblen'], max_agents=self.tf.max_agents)
            # TODO: In future add ViT here as an option
        else:
            self.tf = core.DiffusionTransformer(self.hp_dict, self.hp_dict['diff_exp_name'])
            self.tf_target = deepcopy(self.tf)

            self.tf = self.tf.to(self.device)
            self.tf_target = self.tf_target.to(self.device)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.tf_target.parameters():
                p.requires_grad = False
            self.ma_replay_buffer = MARB.MADiffTD3ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['rblen'], max_agents=self.tf.max_agents, diff_steps=self.diff_timesteps)

        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic1, self.tf.decoder_critic2])
        print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        self.actor_params = self.tf.decoder_actor.parameters()
        self.critic_params = itertools.chain(self.tf.decoder_critic1.parameters(), self.tf.decoder_critic2.parameters())
        
        self.optimizer_actor = optim.AdamW(filter(lambda p: p.requires_grad, self.actor_params), lr=hp_dict['pi_lr'], weight_decay=1e-2)
        self.optimizer_critic = optim.AdamW(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'], weight_decay=1e-2)

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=20, T_mult=2, eta_min=hp_dict['eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=20, T_mult=2, eta_min=hp_dict['eta_min'])
        
        if self.hp_dict['diff_exp_name']=="expt_1":
            self.q_loss_fn = loss_utils.matsac_q_loss
        elif self.hp_dict['diff_exp_name']=="expt_2":
            self.q_loss_fn = loss_utils.matsac_q_loss_diff
            
        self.q_loss = None
        self.internal_updates_counter = 0
        
        self.q_loss_scaler = torch.amp.GradScaler('cuda')
        self.pi_loss_scaler = torch.amp.GradScaler('cuda')

    def w_k_decay(self, k):
        return self.w_k**k

    def update_Q(self, s1, a, s2, r, d, pos):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ** q dim: bs, N, 1  (with mean(1) -> bs, 1); rew dim after unsqueeze: bs, 1
            # ** By unsqueezing the reward we compute a decentralized Q-value weightage in the loss
            q1 = self.tf.decoder_critic1(s1, a, pos).squeeze().mean(dim=1)
            q2 = self.tf.decoder_critic2(s1, a, pos).squeeze().mean(dim=1)
            with torch.no_grad():
                """ For a Bandit Problem setting, all dones are True and hence, q_next only becomes reward """
                # TODO Commented to save compute. For long horizon tasks, uncomment this
                # bs, N, _ = s2.size()
                # a_K = torch.randn((bs, N, self.act_dim), device=self.device)
                # next_actions = self.tf.get_actions(a, s2, pos, get_low_level=False)
                
                # next_q1 = self.tf_target.decoder_critic1(s2, next_actions, pos).squeeze()
                # next_q2 = self.tf_target.decoder_critic2(s2, next_actions, pos).squeeze()
                # q_next = r + self.gamma * ((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2))).mean(dim=1)
                q_next = r

            q_loss1 = F.mse_loss(q1, q_next)
            q_loss2 = F.mse_loss(q2, q_next)
            
        self.q_loss_scaler.scale(q_loss1).backward()
        self.q_loss_scaler.scale(q_loss2).backward()
        
        self.q_loss_scaler.unscale_(self.optimizer_critic)
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.hp_dict['max_grad_norm'])
        
        self.q_loss_scaler.step(self.optimizer_critic)
        self.q_loss_scaler.update()
        
        self.logger.add_data('Q loss', q_loss1.item() + q_loss2.item())

    def update_pi(self, k, s1, a_0, a_k, log_p_old, pos):
        for p in self.critic_params:
            p.requires_grad = False
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a_k_minus_1, log_p, H = self.tf.get_actions_k_minus_1(k, a_k, s1, pos)
            
            q1_pi = self.tf_target.decoder_critic1(s1, a_0, pos).squeeze() #.mean(dim=1)
            q2_pi = self.tf_target.decoder_critic2(s1, a_0, pos).squeeze() #.mean(dim=1)
            q_pi = torch.minimum(q1_pi, q2_pi)
            
            log_p = log_p #.sum(dim=(-2, -1))
            log_p_old = log_p_old #.sum(dim=(-2, -1))
            imp_weights = torch.exp(log_p - log_p_old)
            
            # TODO: measure the ratio value and see if it is improving over time
            
            surr1 = imp_weights * q_pi * self.w_k_decay(k)
            surr2 = torch.clamp(imp_weights, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * q_pi
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            
            # !! No entropy term for now. If we just add this, we'll get SAC! :D
            pi_loss = policy_loss # + self.H_coeff * H 
            
        self.pi_loss_scaler.scale(pi_loss).backward()
        self.pi_loss_scaler.unscale_(self.optimizer_actor)
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.hp_dict['max_grad_norm'])
        self.pi_loss_scaler.step(self.optimizer_actor)
        self.pi_loss_scaler.update()
        
        self.logger.add_data('Pi loss', pi_loss.item())
        self.logger.add_data('Log Prob Ratio', imp_weights.mean().item())
            
        for p in self.critic_params:
            p.requires_grad = True
            
    def sample_low_level_MDP_data(self, states, pos):
        rb_low_level = []
        with torch.no_grad():
            bs, N, _ = states.size()
            a_k = torch.randn((bs, N, self.act_dim), device=self.device)
            for k in range(self.diff_timesteps-1, 0, -1):
                a_k, log_p_k, H = self.tf.get_actions_k_minus_1(k, a_k, states, pos)
                rb_low_level.append([deepcopy(a_k), deepcopy(log_p_k), k])
        return rb_low_level
            
    def update(self, current_episode, n_updates, log_reward):
        for reward in log_reward:
            self.logger.add_data('Reward', reward)
        for j in range(n_updates):
            self.internal_updates_counter += 1
            # if self.internal_updates_counter == 1:
            #     for param_group in self.optimizer_critic.param_groups:
            #         param_group['lr'] = 1e-6
            #     for param_group in self.optimizer_actor.param_groups:
            #         param_group['lr'] = 1e-6
            # elif self.internal_updates_counter == self.hp_dict['warmup_epochs']:
            #     for param_group in self.optimizer_critic.param_groups:
            #         param_group['lr'] = self.hp_dict['q_lr']
            #     for param_group in self.optimizer_actor.param_groups:
            #         param_group['lr'] = self.hp_dict['pi_lr']

            data = self.ma_replay_buffer.sample_batch(self.batch_size)
            n_agents = int(torch.max(data['num_agents']))
            states = data['obs'][:,:n_agents].to(self.device)
            actions = data['act'][:,:n_agents].to(self.device)
            rews = data['rew'].to(self.device)
            new_states = data['obs2'][:,:n_agents].to(self.device)
            dones = data['done'].to(self.device)
            pos = data['pos'][:,:n_agents].to(self.device)

            """ These are of size bs, diff_steps, N, act_dim """
            old_log_ps = data['log_ps'][:,:,:n_agents].to(self.device)
            a_ks = data['a_ks'][:,:,:n_agents].to(self.device)
            # rb_low_level = self.sample_low_level_MDP_data(states, pos)
            
            # Critic Update
            # self.optimizer_critic.zero_grad()
            self.scheduler_critic.step()
            
            """ We are not training the OG Critic on the intermediate values of the low-level MDP """
            # for a_k, log_p, k in random.sample(rb_low_level, self.diff_timesteps//10):
            for param in self.critic_params:
                param.grad = None
            self.update_Q(states, actions, new_states, rews, dones, pos)

            # Actor Update
            # self.optimizer_actor.zero_grad()
            if self.internal_updates_counter % self.policy_delay == 0:
                self.scheduler_actor.step()
                # for a_k, log_p, k in random.sample(rb_low_level, self.diff_timesteps//5):
                
                # TODO: How many times to update the policy? Hypeparam...
                for _ in range(1):
                    shuffled_indices = torch.randperm(self.k_thresh, device=self.device)
                    for i in range(self.k_thresh):
                        k = shuffled_indices[i]
                        for param in self.actor_params:
                            param.grad = None
                        self.update_pi(k, states, actions, a_ks[:,k,:,:], old_log_ps[:,k,:], pos)
                        
                # Target Update
                with torch.no_grad():
                    for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                        p_target.data.mul_(self.hp_dict['tau'])
                        p_target.data.add_((1 - self.hp_dict['tau']) * p.data)
                        
            if self.internal_updates_counter % 1000 == 0:
                print("ckpt saved @ ", current_episode, self.internal_updates_counter)
                dicc = {
                    'model': self.tf.state_dict(),
                    'actor_optimizer': self.optimizer_actor.state_dict(),
                    'critic_optimizer': self.optimizer_critic.state_dict(),
                }
                torch.save(dicc, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")

        self.ma_replay_buffer.update_sample_ptr()
        if self.internal_updates_counter % self.infer_every == 0:
            self.logger.log_metrics(max_length=500)
                
    @torch.no_grad()
    def get_actions(self, obs, pos):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).to(self.device)
        bs, N, _ = obs.size()
        a_K = torch.randn((bs, N, self.act_dim), device=self.device)
        
        actions, a_ks, log_ps, ents = self.tf.get_actions(a_K, obs, pos, get_low_level=True)
        return actions, a_ks, log_ps, ents
        
    def load_saved_policy(self, path):
        dicc = torch.load(path, map_location=self.hp_dict['dev_rl'])
        
        self.tf.load_state_dict(dicc['model'])
        # self.optimizer_actor.load_state_dict(dicc['actor_optimizer'])
        # self.optimizer_critic.load_state_dict(dicc['critic_optimizer'])
        # self.uuid = dicc['uuid']
        self.tf_target = deepcopy(self.tf)