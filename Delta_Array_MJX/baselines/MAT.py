from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import baselines.mappo_network as Actor, Critic
import baselines.multi_agent_replay_buffer as MARB
import baselines.utils as utils

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class R_MAPPO:
    def __init__(self, env_dict, hp_dict):
        self.env_dict = env_dict
        self.hp_dict = hp_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.log_dict = {
            'Q loss': [],
            'Pi loss': [],
            'alpha': [],
            'mu': [],
            'std': [],
        }
        self.pi_lr = self.hp_dict['pi_lr']
        self.q_lr = self.hp_dict['q_lr']
        
        self.actor = Actor(hp_dict['obs_dim'], hp_dict['act_dim'], hp_dict['hidden_dim'], self.device)
        self.critic = Critic(hp_dict['obs_dim'], hp_dict['act_dim'], hp_dict['hidden_dim'], self.device)
        
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.max_agents)
        print("Model Params Size: ", tuple(count_vars(module) for module in [self.actor, self.critic]))
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr, eps=1e-5,)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.q_lr, eps=1e-5,)
        
    def lr_decay(self, episode, episodes):
        utils.update_linear_schedule(self.actor_optimizer, episode, episodes, self.pi_lr)
        utils.update_linear_schedule(self.critic_optimizer, episode, episodes, self.q_lr)
        
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor