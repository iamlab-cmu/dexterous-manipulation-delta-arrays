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

import utils.MABC.dit_core as core
# from utils.openai_utils.logx import EpochLogger
import utils.multi_agent_replay_buffer as MARB

class MADiTSAC:
    def __init__(self, parent_hp_dict, finetune=False):
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
        self.tf = core.DiffusionTransformer(self.hp_dict, mfrl=True)
        self.tf = self.tf.to(self.device)
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.hp_dict['replay_size'], max_agents=self.tf.max_agents)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic])
        print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        self.critic_params = itertools.chain(self.tf.decoder_critic1.parameters(), self.tf.decoder_critic2.parameters())
        
        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'])
            self.optimizer_critic = optim.Adam(filter(lambda p: p.requires_grad, self.critic_params), lr=self.hp_dict['q_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=self.hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.critic_params), lr=self.hp_dict['q_lr'])

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=2, T_mult=2, eta_min=self.hp_dict['pi_eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=2, T_mult=2, eta_min=self.hp_dict['q_eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0
        
        with open('./utils/MADP/normalizer_bc.pkl', 'rb') as f:
            normalizer = pkl.load(f)
        self.obj_name_encoder = normalizer['obj_name_encoder']

    def compute_q_loss(self, s1, a, s2, r, d, obj_name_encs, pos, n_agents):
        # q1 = self.tf.get_q_values(q_values, s1, a, obj_name_encs, pos).mean(dim=1)
        q_next = r.unsqueeze(1)
        q_loss = self.tf.compute_q_loss(self, q_next, s1, a, obj_name_encs, pos)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()

        if not self.hp_dict["dont_log"]:
            self.q_loss = q_loss.cpu().detach().numpy()
            wandb.log({"Q loss":self.q_loss})

    def compute_pi_loss(self, s1, obj_name_encs, pos, n_agents):
        for p in self.model.decoder_critic.parameters():
            p.requires_grad = False

        x_T = torch.randn((1, n_agents, 2), device=self.hp_dict['device'])
        # TODO: CHECK if this works. I doubt it cos generating computation graph on the denoising loop might not be possible. 
        actions = self.tf.get_actions_mfrl(x_T, s1, obj_name_encs, pos)
        
        q_x_T = torch.randn((1, n_agents, 1), device=self.hp_dict['device'])
        q_pi = self.tf.get_q_values(q_x_T, s1, actions, obj_name_encs, pos)
        pi_loss = -q_pi.mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_actor.step()

        if not self.hp_dict["dont_log"]:
            wandb.log({"Pi loss":pi_loss.cpu().detach().numpy()})
        
        for p in self.model.decoder_critic.parameters():
            p.requires_grad = True

    def update(self, batch_size, current_episode, n_envs):
        for _ in range(n_envs):
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
            self.compute_q_loss(states, actions, new_states, rews, dones, obj_name_encs, pos, n_agents)

            # Actor Update
            # with torch.autograd.set_detect_anomaly(True):
            self.optimizer_actor.zero_grad()
            self.compute_pi_loss(states, obj_name_encs, pos, n_agents)

            # Target Update
            # with torch.no_grad():
            #     for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
            #         p_target.data.mul_(self.hp_dict['tau'])
            #         p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

        if (self.train_or_test == "train") and (current_episode % 5000) == 0:
            torch.save(self.tf.state_dict(), f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
    
    @torch.no_grad()
    def get_actions(self, obs, pos, obj_name, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        obj_name_encs = torch.as_tensor(self.obj_name_encoder.transform(np.array(obj_name).ravel()), dtype=torch.int32).to(self.device)
        
        actions = self.tf.get_actions(obs, obj_name_encs, pos, deterministic=deterministic)
        return actions.detach().cpu().numpy()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        self.tf.load_state_dict(torch.load(path, map_location=self.hp_dict['dev_rl'], weights_only=True))
        # self.tf_target = deepcopy(self.tf)