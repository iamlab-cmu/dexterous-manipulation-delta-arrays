import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import itertools
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils.MATSAC.gpt_core as core
# from utils.openai_utils.logx import EpochLogger
from utils.MATSAC.multi_agent_replay_buffer import MultiAgentReplayBuffer

class MATSAC:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), train_or_test="train"):
        # if train_or_test == "train":
            # self.logger = EpochLogger(**logger_kwargs)
            # self.logger.save_config(locals())

        self.train_or_test = train_or_test
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']

        # n_layers_dict={'encoder': 2, 'actor': 2, 'critic': 2}
        self.tf = core.Transformer(self.obs_dim, self.act_dim, self.act_limit, self.hp_dict["model_dim"], self.hp_dict["num_heads"], self.hp_dict["dim_ff"], self.hp_dict["n_layers_dict"], self.hp_dict["dropout"], self.device, self.hp_dict["delta_array_size"], self.hp_dict["adaln"])
        self.tf_target = deepcopy(self.tf)

        self.tf = self.tf.to(self.device)
        self.tf_target = self.tf_target.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.tf_target.parameters():
            p.requires_grad = False

        self.ma_replay_buffer = MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.tf.max_agents)

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
        # Set up model saving
        # if self.train_or_test == "train":
        #     self.logger.setup_pytorch_saver(self.tf)

    def compute_q_loss(self, s1, a, s2, r, d, pos):
        q1 = self.tf.decoder_critic1(s1, a, pos).mean(dim=1)
        q2 = self.tf.decoder_critic2(s1, a, pos).mean(dim=1)
        
        with torch.no_grad():
            # next_state_enc = self.tf.encoder(s2)
            # next_actions, next_log_pi = self.tf.get_actions(next_state_enc)
            """ For now our problem is a single-step problem, so we don't need to compute the next_q values. 
            TODO: Investigate if we can improve something here later. e.g. take inspiration from PPO MAT code and see if I can include entropy and shiiz to add to the q_loss"""
            # next_q1 = self.tf_target.decoder_critic1(next_state_enc, next_actions)
            # next_q2 = self.tf_target.decoder_critic2(next_state_enc, next_actions)
            # q_next = r + self.hp_dict['gamma'] * (1 - d) * (torch.min(next_q1, next_q2) - self.hp_dict['alpha'] * next_log_pi)
            q_next = r.unsqueeze(1)
        q_loss1 = F.mse_loss(q1, q_next)
        q_loss1.backward()        
        q_loss2 = F.mse_loss(q2, q_next)
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic1.parameters(), self.hp_dict['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic2.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()

        if not self.hp_dict["dont_log"]:
            self.q_loss = q_loss1.cpu().detach().numpy() + q_loss2.cpu().detach().numpy()
            wandb.log({"Q loss":self.q_loss})

    def compute_pi_loss(self, s1, current_episode, pos):
        for p in self.critic_params:
            p.requires_grad = False

        actions = self.tf.get_actions(s1, pos)
        # actions = self.tf.get_actions(s1, pos)
        
        q1_pi = self.tf.decoder_critic1(s1, actions, pos)
        q2_pi = self.tf.decoder_critic2(s1, actions, pos)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = -q_pi.mean() #self.hp_dict['alpha'] * logp_pi 

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_actor.step()

        if not self.hp_dict["dont_log"]:
            wandb.log({"Pi loss":pi_loss.cpu().detach().numpy()})
        
        for p in self.critic_params:
            p.requires_grad = True

    def update(self, batch_size, current_episode):
        self.internal_updates_counter += 1
        if self.internal_updates_counter == 1:
            for param_group in self.optimizer_critic.param_groups:
                param_group['lr'] = 1e-6
            for param_group in self.optimizer_actor.param_groups:
                param_group['lr'] = 1e-6
        elif self.internal_updates_counter == 10001:
            for param_group in self.optimizer_critic.param_groups:
                param_group['lr'] = self.hp_dict['q_lr']
            for param_group in self.optimizer_actor.param_groups:
                param_group['lr'] = self.hp_dict['pi_lr']

        data = self.ma_replay_buffer.sample_batch(batch_size)
        n_agents = int(torch.max(data['num_agents']))
        states, actions, rews, new_states, dones = data['obs'][:,:n_agents].to(self.device), data['act'][:,:n_agents].to(self.device), data['rew'].to(self.device), data['obs2'][:,:n_agents].to(self.device), data['done'].to(self.device)
        pos = data['pos'][:,:n_agents].type(torch.int64).to(self.device)

        # Critic Update
        self.optimizer_critic.zero_grad()
        self.compute_q_loss(states, actions, new_states, rews, dones, pos)

        # Actor Update
        # with torch.autograd.set_detect_anomaly(True):
        self.optimizer_actor.zero_grad()
        self.compute_pi_loss(states, current_episode, pos)

        # Target Update
        with torch.no_grad():
            for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                p_target.data.mul_(self.hp_dict['tau'])
                p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

        if (self.train_or_test == "train") and (current_episode % 5000) == 0:
            torch.save(self.tf.state_dict(), f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
    
    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int64).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # actions, _ = self.tf.get_actions_gauss(state_enc, deterministic=deterministic)
            actions = self.tf.get_actions(obs, pos, deterministic=deterministic)
            return actions.detach().cpu().numpy()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        self.tf.load_state_dict(torch.load(path, map_location=self.hp_dict['dev_rl']))
        # self.tf_target = deepcopy(self.tf)