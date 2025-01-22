from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

import baselines.mat_network as core
import Delta_Array_MJX.baselines.multi_agent_ppo_replay_buffer as MARB
import baselines.utils as utils
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MATPPO:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), train_or_test="train"):
        self.train_or_test = train_or_test
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.gauss = hp_dict['gauss']
        self.log_dict = {
            'Q loss': [],
            'Pi loss': [],
            'alpha': [],
            'mu': [],
            'std': [],
        }
        self.max_avg_rew = 0

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
        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic])
        if self.train_or_test == "train":
            print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="adamW":
            self.optimizer_actor = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=hp_dict['q_lr'])

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=10, T_mult=2, eta_min=hp_dict['eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=10, T_mult=2, eta_min=hp_dict['eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0
        
        self.q_loss_scaler = torch.amp.GradScaler('cuda')
        self.pi_loss_scaler = torch.amp.GradScaler('cuda')

        if self.gauss:
            if self.hp_dict['learned_alpha']:
                self.log_alpha = torch.tensor([-1.6094], requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp_dict['pi_lr'])
            else:
                self.alpha = torch.tensor([self.hp_dict['alpha']], requires_grad=False, device=self.device)

    def compute_q_loss(self, s1, a, s2, r, d, pos):
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q = self.tf.decoder_critic(s1, a, pos).squeeze().mean(dim=1)
            with torch.no_grad():
                q_next = r
            q_loss = F.mse_loss(q, q_next)
            
        self.q_loss_scaler.scale(q_loss).backward()
        
        self.q_loss_scaler.unscale_(self.optimizer_critic)
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic.parameters(), self.hp_dict['max_grad_norm'])
        
        self.q_loss_scaler.step(self.optimizer_critic)
        self.q_loss_scaler.update()
        
        self.log_dict['Q loss'].append(q_loss.item())

    def compute_pi_loss(self, s1, pos):
        for p in self.tf.decoder_critic.parameters():
            p.requires_grad = False
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, n_agents, _ = s1.size()
            if self.gauss:
                actions, log_probs, mu, std = self.tf(s1, pos)
            else:
                actions = self.tf(s1, pos)
                
            q_pi = self.tf.decoder_critic(s1, actions, pos).squeeze()
            
            if self.gauss:
                pi_loss = (self.alpha * log_probs - q_pi).mean()
            else:
                pi_loss = -q_pi.mean()
            
        self.pi_loss_scaler.scale(pi_loss).backward()
        self.pi_loss_scaler.unscale_(self.optimizer_actor)
        # pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.pi_loss_scaler.step(self.optimizer_actor)
        self.pi_loss_scaler.update()
        
        # Update alpha
        if self.hp_dict['learned_alpha']:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_probs - self.act_dim*n_agents).detach()).mean() # Target entropy is -act_dim
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        for p in self.tf.decoder_critic.parameters():
            p.requires_grad = True
            
        self.log_dict['Pi loss'].append(pi_loss.item())
        if self.gauss:
            self.log_dict['alpha'].append(self.alpha.item())
            self.log_dict['mu'].append(mu.mean().item())
            self.log_dict['std'].append(std.mean().item())

    def update(self, batch_size, current_episode, n_updates, logged_rew):
        wandb.log({'Reward': logged_rew})
        for j in range(n_updates):
            self.internal_updates_counter += 1
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

            for param in self.tf.decoder_critic.parameters():
                param.grad = None
            self.compute_q_loss(states, actions, new_states, rews, dones, pos)

            for param in self.tf.decoder_actor.parameters():
                param.grad = None
            self.compute_pi_loss(states, pos)

            # Target Update
            with torch.no_grad():
                for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                    p_target.data.mul_(self.hp_dict['tau'])
                    p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

            if (self.train_or_test == "train") and (self.internal_updates_counter % 5000) == 0:
                if self.max_avg_rew < logged_rew:
                    print("ckpt saved @ ", current_episode, self.internal_updates_counter)
                    self.max_avg_rew = logged_rew
                    dicc = {
                        'model': self.tf.state_dict(),
                        'actor_optimizer': self.optimizer_actor.state_dict(),
                        'critic_optimizer': self.optimizer_critic.state_dict(),
                        'uuid': self.uuid
                    }
                    torch.save(dicc, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
        
            if (not self.hp_dict["dont_log"]) and (self.internal_updates_counter % 100) == 0:
                wandb.log({k: np.mean(v) if isinstance(v, list) and len(v) > 0 else v for k, v in self.log_dict.items()})
                self.log_dict = {
                    'Q loss': [],
                    'Pi loss': [],
                    'alpha': [],
                    'mu': [],
                    'std': [],
                }
                
    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        
        actions = self.tf.get_actions(obs, pos, deterministic=deterministic)
        return actions.tolist()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        dicc = torch.load(path, map_location=self.hp_dict['dev_rl'])
        
        self.tf.load_state_dict(dicc['model'])
        # self.optimizer_actor.load_state_dict(dicc['actor_optimizer'])
        # self.optimizer_critic.load_state_dict(dicc['critic_optimizer'])
        self.uuid = dicc['uuid']
        self.tf_target = deepcopy(self.tf)