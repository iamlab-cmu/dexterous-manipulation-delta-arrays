import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tqdm
import itertools
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# from utils.MADP.dit_core import DiffusionTransformer, EMA
from utils.MADP.gpt_adaln_no_autoreg import Transformer, count_vars
import utils.multi_agent_replay_buffer as MARB

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

rb_pos_world = np.zeros((8,8,2))
kdtree_positions_world = np.zeros((64, 2))
for i in range(8):
    for j in range(8):
        if i%2!=0:
            finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
        else:
            finger_pos = np.array((i*0.0375, j*0.043301))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
        kdtree_positions_world[i*8 + j, :] = rb_pos_world[i,j]

class MABC_Finetune:
    def __init__(self, parent_hp_dict):
        self.hp_dict = {
            "exp_name"          : parent_hp_dict['exp_name'],
            "data_dir"          : "./data/rl_data",
            "ckpt_loc"          : "./utils/MADP/mabc_new_data_ac_gauss.pth",
            "dont_log"          : parent_hp_dict['dont_log'],
            "replay_size"       : 500001,
            'warmup_epochs'     : 1000,
            'pi_lr'             : parent_hp_dict['pi_lr'],
            'q_lr'              : parent_hp_dict['q_lr'],
            "q_eta_min"         : parent_hp_dict['q_eta_min'],
            "pi_eta_min"        : parent_hp_dict['pi_eta_min'],
            'ckpt_dir'          : './mabc_finetune_final.pth',
            'idx_embed_loc'     : './utils/MADP/idx_embedding_new.pth',
            "tau"               : 0.005,
            "gamma"             : 0.99,

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            'act_limit'         : 0.03,
            "device"            : parent_hp_dict['dev_rl'],
            'optim'             : 'adam',
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : parent_hp_dict['max_grad_norm'],
            "alpha"             : 0.2,
            "adaln"             : parent_hp_dict['adaln'],
            'masked'            : parent_hp_dict['masked'],
            'cmu_ri'            : parent_hp_dict['cmu_ri'],
            'gauss'             : parent_hp_dict['gauss'],
            'ca'                : parent_hp_dict['ca'],
            'learned_alpha'     : parent_hp_dict['learned_alpha'],
        }
        self.device = self.hp_dict['device']
        self.tf = Transformer(self.hp_dict)
        self.gauss = self.hp_dict['gauss']
        # self.tf.to(self.hp_dict['device'])
        # self.optimizer = optim.AdamW(self.tf.parameters(), lr=self.hp_dict['pi_lr'], weight_decay=0)
        # self.tf.load_state_dict(torch.load(self.hp_dict['ckpt_loc'], weights_only=False)['tf'])
        
        self.tf.to(self.hp_dict['device'])
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.hp_dict['state_dim'], act_dim=self.hp_dict['action_dim'], size=self.hp_dict['replay_size'], max_agents=self.tf.max_agents)
        
        var_counts = tuple(count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic])
        # if self.train_or_test == "train":
        print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'], weight_decay=1e-6)
            self.optimizer_critic = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=self.hp_dict['q_lr'], weight_decay=1e-6)
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=self.hp_dict['q_lr'])
        else:
            raise NotImplementedError

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=20, T_mult=2, eta_min=self.hp_dict['pi_eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=20, T_mult=2, eta_min=self.hp_dict['q_eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0
        
        if self.gauss:
            if self.hp_dict['learned_alpha']:
                self.log_alpha = torch.tensor([-1.6094], requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.hp_dict['pi_lr'])
            else:
                self.alpha = torch.tensor([0.2], requires_grad=False, device=self.device)
        
    def compute_q_loss(self, s1, a, s2, r, d, pos):
        q = self.tf.decoder_critic(s1, a, pos).squeeze().mean(dim=1)
        
        with torch.no_grad():
            if self.gauss:
                next_actions, log_probs, _, _= self.tf(s2, pos)
            else:
                next_actions = self.tf(s2, pos)
            
            next_q = self.tf.decoder_critic(s2, next_actions, pos).squeeze()
            q_next = r + self.hp_dict['gamma'] * ((1 - d.unsqueeze(1)) * (next_q - self.alpha * log_probs)).mean(dim=1)
            # q_next = r.unsqueeze(1)
        q_loss = F.mse_loss(q, q_next)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()
        
        self.log_dict['Q loss'].append(q_loss.item())

    def compute_pi_loss(self, s1, pos):
        for p in self.tf.decoder_critic.parameters():
            p.requires_grad = False
        
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
        
        for p in self.tf.decoder_critic.parameters():
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
            if self.internal_updates_counter == 1:
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.hp_dict['pi_eta_min']
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.hp_dict['pi_eta_min']
            elif self.internal_updates_counter == self.hp_dict['warmup_epochs']:
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.hp_dict['q_lr']
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.hp_dict['pi_lr']
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
            self.optimizer_actor.zero_grad()
            self.compute_pi_loss(states, pos)

            # Target Update
            # with torch.no_grad():
            #     for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
            #         p_target.data.mul_(self.hp_dict['tau'])
            #         p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

        if not self.hp_dict["dont_log"]:
            wandb.log({k: np.mean(v) if isinstance(v, list) and len(v) > 0 else v for k, v in self.log_dict.items()})
                
        if self.internal_updates_counter % 50000 == 0:
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
