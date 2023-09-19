import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.SAC.replay_buffer import ReplayBuffer

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size=256, learning_rate=3e-4, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        self.fc4_mean = nn.Linear(hidden_size, n_actions)
        self.fc4_mean.weight.data.uniform_(-init_w, init_w)
        self.fc4_mean.bias.data.uniform_(-init_w, init_w)
        self.fc4_log_std = nn.Linear(hidden_size, n_actions)
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
        # log_pi = log_pi.sum(dim=1, keepdim=True)
        log_pi = log_pi.sum() # for batch_size = 1
        return action, log_pi

class REINFORCE:
    def __init__(self, env_dict, lr, wandb_bool=False):
        self.env_dict = env_dict
        self.action_range = [self.env_dict['action_space']['low'], self.env_dict['action_space']['high']]
        # self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_nw = PolicyNetwork(self.env_dict['observation_space']['dim'], self.env_dict['action_space']['dim'], 256)
        self.replay_buffer = ReplayBuffer(10000)
        
        self.lr = lr
        self.wandb_bool = wandb_bool
        self.expt_no = None
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
        self.expt_no = run
        wandb.init(project="SAC", 
            name=f"experiment_{run}",
            config=hp_dict)

    def end_wandb(self):
        if self.wandb_bool:
            wandb.finish()
            
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

    def rescale_action(self, action):
        return (action + 1)/2 * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
    
    def save_policy_model(self):
        torch.save(self.policy_nw.state_dict(), f"./utils/SAC/policy_models/reinforce_expt_{self.expt_no}.pt")

    def update_policy_reinforce(self, action_log_pi, reward):
        action_log_pi = torch.tensor(action_log_pi.clone().detach().requires_grad_(True))
        reward = torch.tensor(reward.copy(), requires_grad=True)
        
        self.policy_nw.optimizer.zero_grad()
        policy_gradient = -self.lr * action_log_pi * reward
        policy_gradient.backward()
        self.policy_nw.optimizer.step()