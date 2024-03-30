import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils.MATDQN.gpt_core as core
# from utils.openai_utils.logx import EpochLogger
from utils.MATDQN.multi_agent_replay_buffer import MultiAgentReplayBuffer

class MATDQN:
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
        self.tf = core.Transformer(self.obs_dim, self.act_dim, self.act_limit, self.hp_dict["model_dim"], self.hp_dict["num_heads"], self.hp_dict["dim_ff"], self.hp_dict["n_layers_dict"], self.hp_dict["dropout"], self.device, self.hp_dict["delta_array_size"])
        self.tf_target = deepcopy(self.tf)

        self.tf = self.tf.to(self.device)
        self.tf_target = self.tf_target.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.tf_target.parameters():
            p.requires_grad = False

        self.ma_replay_buffer = MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.tf.max_agents)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_critic])
        if self.train_or_test == "train":
            print(f"\nNumber of parameters: {np.sum(var_counts)}\n")
        
        if self.hp_dict['optim']=="adam":
            self.optimizer_critic = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_critic.parameters()), lr=hp_dict['q_lr'])
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=5, T_mult=2, eta_min=hp_dict['eta_min'])
        
        self.q_loss = None
        # Set up model saving
        # if self.train_or_test == "train":
        #     self.logger.setup_pytorch_saver(self.tf)

    def compute_q_loss(self, s1, a, s2, r, d, pos):
        q = self.tf.decoder_critic(s1, a, pos).mean(dim=(1, 2), keepdim=True).squeeze(-1)
        with torch.no_grad():
            # next_state_enc = self.tf.encoder(s2)
            # next_actions, next_log_pi = self.tf.get_actions(next_state_enc)
            """ For now our problem is a single-step problem, so we don't need to compute the next_q values. 
            TODO: Investigate if we can improve something here later. e.g. take inspiration from PPO MAT code and see if I can include entropy and shiiz to add to the q_loss"""
            # next_q1 = self.tf_target.decoder_critic1(next_state_enc, next_actions)
            # next_q2 = self.tf_target.decoder_critic2(next_state_enc, next_actions)
            # q_next = r + self.hp_dict['gamma'] * (1 - d) * (torch.min(next_q1, next_q2) - self.hp_dict['alpha'] * next_log_pi)
            q_next = r.unsqueeze(1)

        q_loss = F.mse_loss(q, q_next)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_critic.parameters(), self.hp_dict['max_grad_norm'])
        self.optimizer_critic.step()

        if not self.hp_dict["dont_log"]:
            self.q_loss = q_loss.cpu().detach().numpy()

    def update(self, batch_size, current_episode):
        self.scheduler.step(current_episode)
        data = self.ma_replay_buffer.sample_batch(batch_size)
        n_agents = int(torch.max(data['num_agents']))
        states, actions, rews, new_states, dones = data['obs'][:,:n_agents].to(self.device), data['act'][:,:n_agents].to(self.device), data['rew'].to(self.device), data['obs2'][:,:n_agents].to(self.device), data['done'].to(self.device)
        pos = data['pos'][:,:n_agents].type(torch.int64).to(self.device)

        # DQN Update
        self.optimizer_critic.zero_grad()
        self.compute_q_loss(states, actions, new_states, rews, dones, pos)

        # Target Update
        with torch.no_grad():
            for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                p_target.data.mul_(self.hp_dict['tau'])
                p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

        if (self.train_or_test == "train") and (current_episode % 5000) == 0:
            torch.save(self.tf.state_dict(), f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
    
    def cross_entropy_method(self, obs, pos, num_agents, iters=5, n_samples=512, top_k=20):
        if num_agents < 2:
            return np.zeros((1, 2))

        x_bounds = np.array([[-0.03, 0.03]] * num_agents)  # Shape: (num_agents, 2)
        y_bounds = np.array([[-0.03, 0.03]] * num_agents)  # Shape: (num_agents, 2)

        for iteration in range(iters):
            x_samples = np.random.uniform(x_bounds[:, 0][:, np.newaxis], x_bounds[:, 1][:, np.newaxis], (n_samples, num_agents, 1))
            y_samples = np.random.uniform(y_bounds[:, 0][:, np.newaxis], y_bounds[:, 1][:, np.newaxis], (n_samples, num_agents, 1))
            samples = np.concatenate((x_samples, y_samples), axis=2)

            act = torch.tensor(samples, dtype=torch.float32).to(self.device)
            q_values = self.tf.decoder_critic(obs, act, pos).detach().cpu().numpy().squeeze()

            # Select top-k samples, vectorized sorting and indexing
            if iteration>=(iters-1):
                top_indices = np.argpartition(q_values, 1, axis=0)[:1] # Since our rewards are negative, we can use this logic.
                top_samples = samples[top_indices, np.arange(num_agents)][0]
                return top_samples
            else:
                top_indices = np.argpartition(q_values, top_k, axis=0)[:top_k] # Since our rewards are negative, we can use this logic.
                top_samples = samples[top_indices, np.arange(num_agents)]

                for agent in range(num_agents):
                    agent_top_samples = top_samples[:, agent, :]  # Shape: (top_k, 2)
                    x_bounds[agent] = [np.min(agent_top_samples[:, 0]), np.max(agent_top_samples[:, 0])]
                    y_bounds[agent] = [np.min(agent_top_samples[:, 1]), np.max(agent_top_samples[:, 1])]

                    # Optionally, adjust the bounds slightly to ensure they don't collapse
                    x_bounds[agent] *= 1.1
                    y_bounds[agent] *= 1.1

    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        n_samples = 256
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).repeat(n_samples, 1, 1).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int64).unsqueeze(0).repeat(n_samples, 1, 1).to(self.device)
        with torch.no_grad():
            actions = self.cross_entropy_method(obs, pos, obs.shape[1], n_samples=n_samples)
        return actions
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        self.tf.load_state_dict(torch.load(path, map_location=self.hp_dict['dev_rl']))
        # self.tf_target = deepcopy(self.tf)