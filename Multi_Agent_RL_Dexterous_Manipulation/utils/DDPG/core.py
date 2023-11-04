import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# class MLPActor(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
#         self.pi = mlp(pi_sizes, activation, nn.Tanh)
#         self.act_limit = act_limit

#     def forward(self, obs):
#         # Return output from network scaled to action space limits.
#         return self.act_limit * self.pi(obs)

# class MLPQFunction(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_dim=128, init_w=3e-3):
        super(MLPActor, self).__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self.act_limit * x
    

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, init_w=3e-3):
        super(MLPQFunction, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x, -1)

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=128):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, act_limit, hidden_sizes)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()