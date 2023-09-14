"""
Code inspired heavily by https://github.com/cyoon1729/Policy-Gradient-Methods/blob/master/sac/sac2019.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.SAC.network import SoftQNetwork, GaussianPolicy
from utils.SAC.replay_buffer import ReplayBuffer
# wandb.login()
import wandb

class SACAgent:
    def __init__(self, env_dict, hp_dict, wandb_bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_dict = env_dict
        self.action_range = [env_dict['action_space']['low'], env_dict['action_space']['high']]
        self.obs_dim = env_dict['observation_space']['dim']
        self.action_dim = env_dict['action_space']['dim']

        self.gamma = hp_dict['gamma']
        self.tau = hp_dict['tau']
        self.update_step = 0
        self.delay_step = 2

        self.Q1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.Q2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_Q1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_Q2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param)
        
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=hp_dict['q_lr'])
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=hp_dict['q_lr'])
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=hp_dict['policy_lr'])

        self.alpha = hp_dict['alpha']
        self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=hp_dict['a_lr'])

        self.replay_buffer = ReplayBuffer(hp_dict['buffer_maxlen'])
        self.wandb_bool = wandb_bool
        if self.wandb_bool:
            self.setup_wandb(hp_dict)

    def setup_wandb(self, hp_dict):
        if os.path.exists("./utils/SAC/runtracker.txt"):
            with open("./utils/SAC/runtracker.txt", "r") as f:
                run = int(f.readline())
                run += 1
        else:
            run = 0
        with open("./utils/SAC/runtracker.txt", "w") as f:
            f.write(str(run))

        wandb.init(project="SAC", 
            name=f"experiment_{run}",
            config=hp_dict)

    def end_wandb(self):
        if self.wandb_bool:
            wandb.finish()

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean, log_std = self.policy.forward(obs)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        return self.rescale_action(action.cpu().detach().squeeze(0).numpy())

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(torch.stack(states)).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(torch.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy.sample(next_states)

        # Do twin delayed DDPG stuff here
        next_q1 = self.target_Q1(next_states, next_actions)
        next_q2 = self.target_Q2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        current_q1 = self.Q1(states, actions)
        current_q2 = self.Q2(states, actions)
        q1_loss = F.mse_loss(current_q1, expected_q.detach())
        q2_loss = F.mse_loss(current_q2, expected_q.detach())

        self.Q1_optimizer.zero_grad()
        q1_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.zero_grad()
        q2_loss.backward()
        self.Q2_optimizer.step()
        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item()
        }

        # Delayed update for polucy and target networks'
        new_actions, log_pi = self.policy.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(self.Q1(states, new_actions), self.Q2(states, new_actions))
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            metrics["policy_loss"] = policy_loss.item()
        if self.wandb_bool:
            wandb.log(metrics)
        # Temperature updates
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

# def train(env, agent, max_episodes, max_steps, batch_size):
#     ep_rewards = []
#     for ep in range(max_episodes):
#         obs, _ = env.reset()
#         ep_reward = 0
#         for step in range(max_steps):
#             action = agent.get_action(obs)
#             next_obs, reward, terminate, truncate, _ = env.step(action)
#             agent.replay_buffer.push(obs, action, reward, next_obs, terminate or truncate)
#             ep_reward += reward
            
#             if len(agent.replay_buffer) > batch_size:
#                 agent.update(batch_size)

#             if terminate or truncate or (step==max_steps-1):
#                 ep_rewards.append(ep_reward)
#                 print("Episode: {}, Reward: {}".format(ep, ep_reward))
#                 print(step)
#                 break
#             obs = next_obs
#         print(step)
#     return ep_rewards


# if __name__=="__main__":
#     env = gym.make("Pendulum-v1")
#     agent = SACAgent(env=env, 
#         gamma=0.99, 
#         tau=0.01, 
#         alpha=0.2, 
#         q_lr=3e-4, 
#         policy_lr=3e-4, 
#         a_lr=3e-4, 
#         buffer_maxlen=1000000
#     )

    # ep_rewards = train(env, agent, 
    #     max_episodes=1000, 
    #     max_steps=1600, 
    #     batch_size=256
    # )
