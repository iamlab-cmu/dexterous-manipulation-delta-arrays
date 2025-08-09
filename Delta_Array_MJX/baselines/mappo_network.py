import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import wt_init_, MLP, RNN, ActLayer, PopArt

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, device):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.fc1 = MLP(obs_dim, hidden_dim, 2)
        self.rnn = RNN(hidden_dim, hidden_dim, 1)
        self.act = ActLayer(hidden_dim, act_dim)
    
    def forward(self, obs, rnn_states, masks, available_actions, deterministic):
        # Do this check and stuff on a higher level code
        # obs = check(obs).to(**self.tpdv)
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)
        
        actor_features = self.fc1(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None):
        actor_features = self.fc1(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_log_probs, dist_entropy
    
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, device):
        super(Critic, self).__init__()
        """ Obs are centralized obs."""
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.fc1 = MLP(obs_dim, hidden_dim, 2)
        self.rnn = RNN(hidden_dim, hidden_dim, 1)
        self.v_out = wt_init_(PopArt(self.hidden_size, 1, device=device))
        
    def forward(self, obs, rnn_states, masks):
        critic_features = self.fc1(obs)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states

# class R_MAPPO(nn.Module):
#     def __init__(self, hp_dict, delta_array_size = (8, 8)):
#         super(R_MAPPO, self).__init__()
#         self.hp_dict = hp_dict
#         self.device = hp_dict['dev_rl']
#         self.max_agents = delta_array_size[0] * delta_array_size[1]
#         self.act_limit = hp_dict['act_limit']
#         self.action_dim = hp_dict['action_dim']
        
        
    