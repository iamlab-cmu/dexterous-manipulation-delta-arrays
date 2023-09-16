import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

GAMMA = 0.9

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size=256, learning_rate=3e-4, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        self.fc4_mean = nn.Linear(hidden_dim, n_actions)
        self.fc4_mean.weight.data.uniform_(-init_w, init_w)
        self.fc4_mean.bias.data.uniform_(-init_w, init_w)
        self.fc4_log_std = nn.Linear(hidden_dim, n_actions)
        self.fc4_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc4_log_std.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean = self.fc4_mean(x)
        log_std = self.fc4_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def get_action(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)
        log_pi = log_pi.sum(dim=1, keepdim=True)
        return action, log_pi

class REINFORCE:
    def __init__(self, env, gamma, lr):
        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_nw = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0], 256)
        self.policy_nw.to(self.device)
        self.replay_buffer = ReplayBuffer(buffer_maxlen)
        
        self.lr = lr
    # def get_action(self, obs):
    #     obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
    #     self.policy.eval()
    #     with torch.no_grad():
    #         mean, log_std = self.policy.forward(obs)
    #     self.policy.train()
    #     std = log_std.exp()

    #     normal = torch.distributions.Normal(mean, std)
    #     z = normal.sample()
    #     action = torch.tanh(z)
    #     return self.rescale_action(action.cpu().detach().squeeze(0).numpy())

    # def rescale_action(self, action):
    #     return (action + 1)/2 * (self.action_range[1] - self.action_range[0]) + self.action_range[0]

    def update_policy(batch_size):
        states, action_log_pis, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        # states = torch.tensor(states).to(self.device)
        action_log_pis = torch.tensor(action_log_pis).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        # next_states = torch.tensor(next_states).to(self.device)
        # dones = torch.tensor(dones).to(self.device)
        # dones = dones.view(dones.size(0), -1)

        # new_actions, log_pi = self.policy.sample(states)
        policy_nw.optimizer.zero_grad()
        policy_gradient = self.lr * -action_log_pis * rewards
        policy_gradient.backward()
        policy_nw.optimizer.step()