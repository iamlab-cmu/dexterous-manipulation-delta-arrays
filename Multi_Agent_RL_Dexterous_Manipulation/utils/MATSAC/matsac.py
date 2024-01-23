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

import utils.MATSAC.core as core
from utils.openai_utils.logx import EpochLogger
from utils.MATSAC.multi_agent_replay_buffer import MultiAgentReplayBuffer

class MATSAC:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), train_or_test="train"):
        if train_or_test == "train":
            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(locals())

        self.train_or_test = train_or_test
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['device']

        # n_layers_dict={'encoder': 2, 'actor': 2, 'critic': 2}
        self.tf = core.Transformer(self.obs_dim, self.act_dim, self.act_limit, self.hp_dict["model_dim"], self.hp_dict["num_heads"], self.hp_dict["dim_ff"], self.hp_dict["n_layers_dict"], self.hp_dict["dropout"], self.device, self.hp_dict["delta_array_size"])
        # self.tf_target = deepcopy(self.tf)

        self.tf = self.tf.to(self.device)
        # self.tf_target = self.tf_target.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        # for p in self.tf_target.parameters():
        #     p.requires_grad = False

        self.ma_replay_buffer = MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.tf.max_agents)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.tf.encoder, self.tf.decoder_actor, self.tf.decoder_critic1, self.tf.decoder_critic2])
        print(var_counts)
        if self.train_or_test == "train":
            self.logger.log('\nNumber of parameters: \t Encoder: %d, \t Actor Decoder: %d, \t Critic Decoder1: %d, \t Critic Decoder2: %d\n'%var_counts)

        self.critic_params = itertools.chain(self.tf.encoder.parameters(), self.tf.decoder_critic1.parameters(), self.tf.decoder_critic2.parameters())
        self.optimizer_critic = optim.Adam(self.critic_params, lr=hp_dict['q_lr'])
        self.optimizer_actor = optim.Adam(self.tf.decoder_actor.parameters(), lr=hp_dict['pi_lr'])

        # Set up model saving
        if self.train_or_test == "train":
            self.logger.setup_pytorch_saver(self.tf)

    def compute_q_loss(self, s1, a, s2, r, d):
        state_enc = self.tf.encoder(s1)
        q1 = self.tf.decoder_critic1(state_enc, a)
        q2 = self.tf.decoder_critic2(state_enc, a)
        
        with torch.no_grad():
            # next_state_enc = self.tf.encoder(s2)
            # next_actions, next_log_pi = self.tf.get_actions(next_state_enc)
            """ For now our problem is a single-step problem, so we don't need to compute the next_q values. 
            TODO: Investigate if we can improve something here later. e.g. take inspiration from PPO MAT code and see if I can include entropy and shiiz to add to the q_loss"""
            # next_q1 = self.tf_target.decoder_critic1(next_state_enc, next_actions)
            # next_q2 = self.tf_target.decoder_critic2(next_state_enc, next_actions)
            # q_next = r + self.hp_dict['gamma'] * (1 - d) * (torch.min(next_q1, next_q2) - self.hp_dict['alpha'] * next_log_pi)
            q_next = r.unsqueeze(1)

        q_loss = F.mse_loss(q1, q_next) + F.mse_loss(q2, q_next)

        q_loss.backward()
        self.optimizer_critic.step()
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy())

        wandb.log({"Q loss":q_loss.cpu().detach().numpy()})
        return q_loss, q_info

    def compute_pi_loss(self, s1):
        with torch.no_grad():
            state_enc = self.tf.encoder(s1)
        actions, logp_pi = self.tf.get_actions(state_enc)
        
        q1_pi = self.tf.decoder_critic1(state_enc, actions)
        q2_pi = self.tf.decoder_critic2(state_enc, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.hp_dict['alpha'] * logp_pi - q_pi).mean()

        pi_loss.backward()
        self.optimizer_actor.step()
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        wandb.log({"Pi loss":pi_loss.cpu().detach().numpy()})
        return pi_loss, pi_info

    def update(self, batch_size):
        data = self.ma_replay_buffer.sample_batch(batch_size)
        n_agents = int(torch.max(data['num_agents']))
        # print(n_agents, n_agents[0])
        states, actions, rews, new_states, dones = data['obs'][:,:n_agents].to(self.device), data['act'][:,:n_agents].to(self.device), data['rew'].to(self.device), data['obs2'][:,:n_agents].to(self.device), data['done'].to(self.device)

        # Critic Update
        self.optimizer_critic.zero_grad()
        q_loss, q_info = self.compute_q_loss(states, actions, new_states, rews, dones)
        # self.logger.store(LossQ=q_loss.item(), **q_info)
        
        # Set action embeddings of critic to be the same as actor so there is consistency.
        # self.tf.decoder_critic1.action_embedding.weight.data = self.tf.decoder_actor.action_embedding.weight.data.clone()
        # self.tf.decoder_critic1.action_embedding.bias.data = self.tf.decoder_actor.action_embedding.bias.data.clone()
        # self.tf.decoder_critic2.action_embedding.weight.data = self.tf.decoder_actor.action_embedding.weight.data.clone()
        # self.tf.decoder_critic2.action_embedding.bias.data = self.tf.decoder_actor.action_embedding.bias.data.clone()

        # Actor Update
        for p in self.critic_params:
            p.requires_grad = False

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer_actor.zero_grad()
            pi_loss, pi_info = self.compute_pi_loss(states)
            # self.logger.store(LossPi=pi_loss.item(), **pi_info)

        for p in self.critic_params:
            p.requires_grad = True

        # Target Update
        # with torch.no_grad():
        #     for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
        #         p_target.data.mul_(self.hp_dict['tau'])
        #         p_target.data.add_((1 - self.hp_dict['tau']) * p.data)
    
    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            state_enc = self.tf.encoder(obs)
            actions, _ = self.tf.get_actions(state_enc, deterministic=deterministic)
            return actions.detach().cpu().numpy()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        self.tf.load_state_dict(torch.load(path))
        # self.tf_target = deepcopy(self.tf)