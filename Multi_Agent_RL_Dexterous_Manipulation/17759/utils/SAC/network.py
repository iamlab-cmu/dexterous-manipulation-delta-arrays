import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SoftQNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_dim=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(n_obs + n_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GaussianPolicy(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_dim=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(n_obs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc3_mean = nn.Linear(hidden_dim, n_actions)
        self.fc3_mean.weight.data.uniform_(-init_w, init_w)
        self.fc3_mean.bias.data.uniform_(-init_w, init_w)
        
        self.fc3_log_std = nn.Linear(hidden_dim, n_actions)
        self.fc3_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc3_log_std.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.fc3_mean(x)
        log_std = self.fc3_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)
        log_pi = log_pi.sum(dim=1, keepdim=True)
        return action, log_pi
