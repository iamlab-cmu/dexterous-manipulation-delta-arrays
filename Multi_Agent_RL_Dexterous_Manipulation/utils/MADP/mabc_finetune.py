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
from utils.MADP.gpt_adaln_core import Transformer, count_vars
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
            'ckpt_dir'          : './matil_expt_1.pth',
            'idx_embed_loc'     : './utils/MADP/idx_embedding_128.pth',

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            'act_limit'         : 0.03,
            "device"            : parent_hp_dict['dev_rl'],
            'optim'             : 'adam',
            "model_dim"         : 128,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'decoder': 8},
            "dropout"           : 0,
            "max_grad_norm"     : parent_hp_dict['max_grad_norm'],
            "alpha"             : 0.2,
        }
        self.device = self.hp_dict['device']
        self.model = Transformer(self.hp_dict)
        # self.model.to(self.hp_dict['device'])
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hp_dict['pi_lr'], weight_decay=0)
        # self.model.load_state_dict(torch.load(self.hp_dict['ckpt_loc'], weights_only=False)['model'])
        
        self.model.to(self.hp_dict['device'])
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.hp_dict['state_dim'], act_dim=self.hp_dict['action_dim'], size=self.hp_dict['replay_size'], max_agents=self.model.max_agents)
        
        var_counts = tuple(count_vars(module) for module in [self.model.decoder_actor, self.model.decoder_critic])
        # if self.train_or_test == "train":
        print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.AdamW(filter(lambda p: p.requires_grad, self.model.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'], weight_decay=0)
            self.optimizer_critic = optim.AdamW(filter(lambda p: p.requires_grad, self.model.decoder_critic.parameters()), lr=self.hp_dict['q_lr'], weight_decay=0)
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.model.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.model.decoder_critic.parameters()), lr=self.hp_dict['q_lr'])
        else:
            raise NotImplementedError

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=2, T_mult=2, eta_min=self.hp_dict['pi_eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=2, T_mult=2, eta_min=self.hp_dict['q_eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0
        
        with open('./utils/MADP/normalizer_bc.pkl', 'rb') as f:
            normalizer = pkl.load(f)
        self.obj_name_encoder = normalizer['obj_name_encoder']
        
    def compute_q_loss(self, s1, a, s2, r, d, obj_name_encs, pos):
        q = self.model.decoder_critic(s1, a, obj_name_encs, pos).mean(dim=1)
        q_next = r.unsqueeze(1)
        q_loss = F.mse_loss(q, q_next)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.decoder_critic.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()

        if not self.hp_dict["dont_log"]:
            self.q_loss = q_loss.cpu().detach().numpy()
            wandb.log({"Q loss":self.q_loss})

    def compute_pi_loss(self, s1, obj_name_encs, pos):
        for p in self.model.decoder_critic.parameters():
            p.requires_grad = False

        actions, log_probs = self.model(s1, obj_name_encs, pos)
        q_pi = self.model.decoder_critic(s1, actions, obj_name_encs, pos).mean(dim=1)
        pi_loss = (self.hp_dict['alpha'] * log_probs - q_pi).mean()  

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_actor.step()

        if not self.hp_dict["dont_log"]:
            wandb.log({"Pi loss":pi_loss.cpu().detach().numpy(), "Log Probs":log_probs.cpu().detach().numpy()})
        
        for p in self.model.decoder_critic.parameters():
            p.requires_grad = True

    def update(self, batch_size, current_episode, n_envs):
        for _ in range(5):
            self.internal_updates_counter += 1
            if self.internal_updates_counter == 1:
                print("Warum up training phase")
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = 1e-6
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = 1e-6
                    
            elif self.internal_updates_counter == self.hp_dict['warmup_epochs']:
                print("Normal training phase")
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.hp_dict['q_lr']
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.hp_dict['pi_lr']

            data = self.ma_replay_buffer.sample_batch(batch_size)
            n_agents = int(torch.max(data['num_agents']))
            states = data['obs'][:,:n_agents].to(self.device)
            actions = data['act'][:,:n_agents].to(self.device)
            rews = data['rew'].to(self.device)
            new_states = data['obs2'][:,:n_agents].to(self.device)
            dones = data['done'].to(self.device)
            obj_name_encs = data['obj_name_encs'].to(self.device)
            pos = data['pos'][:,:n_agents].to(self.device)
            
            # Critic Update
            self.optimizer_critic.zero_grad()
            self.compute_q_loss(states, actions, new_states, rews, dones, obj_name_encs, pos)

            # Actor Update
            self.optimizer_actor.zero_grad()
            self.compute_pi_loss(states, obj_name_encs, pos)

            # Target Update
            # with torch.no_grad():
            #     for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
            #         p_target.data.mul_(self.hp_dict['tau'])
            #         p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

            if self.internal_updates_counter % 5000 == 0:
                torch.save({"model": self.model.state_dict()}, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
        
    @torch.no_grad()
    def get_actions(self, obs, pos, obj_name, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        # print(np.array(obj_name).ravel().shape)
        obj_name_enc = torch.as_tensor(self.obj_name_encoder.transform(np.array(obj_name).ravel()), dtype=torch.int32).to(self.device)
            
        # print(obs.size(), pos.size(), obj_name_enc.size())
        # print(obs.dtype, pos.dtype, obj_name_enc.dtype)
        actions, logprob = self.model(obs, obj_name_enc, pos, deterministic=deterministic)
        return actions.detach().cpu().numpy()
        
    def load_saved_policy(self, path):
        print("HAKUNA MATATA")
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(expt_dict['model'])
