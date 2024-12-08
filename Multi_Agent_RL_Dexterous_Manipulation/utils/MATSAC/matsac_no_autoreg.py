import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import itertools
import wandb
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils.MATSAC.gpt_core_no_autoreg as core
# from utils.openai_utils.logx import EpochLogger
import utils.multi_agent_replay_buffer as MARB

class MATSAC:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), train_or_test="train"):
        self.train_or_test = train_or_test
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.gauss = hp_dict['gauss']

        if self.hp_dict['data_type'] == "image":
            self.ma_replay_buffer = MARB.MultiAgentImageReplayBuffer(act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.tf.max_agents)
            # TODO: In future add ViT here as an option
        else:
            # self.tf = core.Transformer(self.obs_dim, self.act_dim, self.act_limit, self.hp_dict["model_dim"], self.hp_dict["num_heads"], self.hp_dict["dim_ff"], self.hp_dict["n_layers_dict"], self.hp_dict["dropout"], self.device, self.hp_dict["delta_array_size"], self.hp_dict["adaln"], self.hp_dict['masked'])
            self.tf = core.Transformer(self.hp_dict)
            self.tf_target = deepcopy(self.tf)

            self.tf = self.tf.to(self.device)
            self.tf_target = self.tf_target.to(self.device)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.tf_target.parameters():
                p.requires_grad = False
            self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.tf.max_agents)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic1, self.tf.decoder_critic2])
        if self.train_or_test == "train":
            print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        self.critic_params = itertools.chain(self.tf.decoder_critic1.parameters(), self.tf.decoder_critic2.parameters())
        
        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.Adam(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'])

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=2, T_mult=2, eta_min=hp_dict['eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=2, T_mult=2, eta_min=hp_dict['eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0

        if self.gauss:
            if self.hp_dict['learned_alpha']:
                self.log_alpha = torch.tensor([-1.6094], requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp_dict['pi_lr'])
            else:
                self.alpha = torch.tensor([0.2], requires_grad=False, device=self.device)

    def compute_q_loss(self, s1, a, s2, r, d, pos):
        q1 = self.tf.decoder_critic1(s1, a, pos).squeeze().mean(dim=1)
        q2 = self.tf.decoder_critic2(s1, a, pos).squeeze().mean(dim=1)
        
        with torch.no_grad():
            if self.gauss:
                next_actions, log_probs, _, _= self.tf(s2, pos)
            else:
                next_actions = self.tf(s2, pos)
            
            next_q1 = self.tf.decoder_critic1(s2, next_actions, pos).squeeze()
            next_q2 = self.tf.decoder_critic2(s2, next_actions, pos).squeeze()
            # print(((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2) - self.alpha * log_probs)).mean(dim=1).shape)
            q_next = r + self.hp_dict['gamma'] * ((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2) - self.alpha * log_probs)).mean(dim=1)
            # q_next = r.unsqueeze(1)
        q_loss1 = F.mse_loss(q1, q_next)
        q_loss1.backward()
        q_loss2 = F.mse_loss(q2, q_next)
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic1.parameters(), self.hp_dict['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic2.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()
        
        self.log_dict['Q loss'].append(q_loss1.item() + q_loss2.item())

    def compute_pi_loss(self, s1, pos):
        for p in self.critic_params:
            p.requires_grad = False
        
        _, n_agents, _ = s1.size()
        if self.gauss:
            actions, log_probs, mu, std = self.tf(s1, pos)
        else:
            actions = self.tf(s1, pos)
        # actions = self.tf.get_actions(s1, pos)
        
        q1_pi = self.tf.decoder_critic1(s1, actions, pos).squeeze()
        q2_pi = self.tf.decoder_critic2(s1, actions, pos).squeeze()
        q_pi = torch.minimum(q1_pi, q2_pi)
        
        if self.gauss:
            pi_loss = (self.alpha * log_probs - q_pi).mean()
        else:
            pi_loss = -q_pi.mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_actor.step()
        
        # Update alpha
        if self.hp_dict['learned_alpha']:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_probs - self.act_dim*n_agents).detach()).mean() # Target entropy is -act_dim
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        for p in self.critic_params:
            p.requires_grad = True
            
        self.log_dict['Pi loss'].append(pi_loss.item())
        if self.gauss:
            self.log_dict['alpha'].append(self.alpha.item())
            self.log_dict['mu'].append(mu.mean().item())
            self.log_dict['std'].append(std.mean().item())

    def update(self, batch_size, current_episode, n_envs, logged_rew):
        self.log_dict = {
            'Q loss': [],
            'Pi loss': [],
            'alpha': [],
            'mu': [],
            'std': [],
            'Reward': logged_rew.tolist()
        }
        for j in range(n_envs):
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
            self.scheduler_actor.step()
            self.scheduler_critic.step()

            data = self.ma_replay_buffer.sample_batch(batch_size)
            n_agents = int(torch.max(data['num_agents']))
            states = data['obs'][:,:n_agents].to(self.device)
            actions = data['act'][:,:n_agents].to(self.device)
            rews = data['rew'].to(self.device)
            new_states = data['obs2'][:,:n_agents].to(self.device)
            dones = data['done'].to(self.device)
            pos = data['pos'][:,:n_agents].to(self.device)

            # Critic Update
            self.optimizer_critic.zero_grad()
            self.compute_q_loss(states, actions, new_states, rews, dones, pos)

            # Actor Update
            # with torch.autograd.set_detect_anomaly(True):
            self.optimizer_actor.zero_grad()
            self.compute_pi_loss(states, pos)

            # Target Update
            with torch.no_grad():
                for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                    p_target.data.mul_(self.hp_dict['tau'])
                    p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

        if not self.hp_dict["dont_log"]:
            wandb.log({k: np.mean(v) if isinstance(v, list) and len(v) > 0 else v for k, v in self.log_dict.items()})
                
        if (self.train_or_test == "train") and (self.internal_updates_counter % 50000) == 0:
            torch.save(self.tf.state_dict(), f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
    
    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        
        actions = self.tf.get_actions(obs, pos, deterministic=deterministic)
        return actions.detach().cpu().numpy()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        print(path)
        self.tf.load_state_dict(torch.load(path, map_location=self.hp_dict['dev_rl']))
        self.tf_target = deepcopy(self.tf)