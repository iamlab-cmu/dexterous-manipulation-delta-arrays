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
        self.clip_param = self.hp_dict['clip_param']
        self.huber_delta = self.hp_dict['huber_delta']
        self.entropy_coef = self.hp_dict['entropy_coef']
        self.max_grad_norm = self.hp_dict['max_grad_norm']
        
        self.actor = Actor(hp_dict['obs_dim'], hp_dict['act_dim'], hp_dict['hidden_dim'], self.device)
        self.critic = Critic(hp_dict['obs_dim'], hp_dict['act_dim'], hp_dict['hidden_dim'], self.device)
        
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'], max_agents=self.max_agents)
        print("Model Params Size: ", tuple(count_vars(module) for module in [self.actor, self.critic]))
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr, eps=1e-5,)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.q_lr, eps=1e-5,)
        
        self.value_normalizer = self.policy.critic.v_out
    
    def calc_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        self.value_normalizer.update(return_batch)
        error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(return_batch) - values
        
        value_loss_clipped = utils.huber_loss(error_clipped, self.huber_delta)
        value_loss_original = utils.huber_loss(error_original, self.huber_delta)
        return torch.max(value_loss_original, value_loss_clipped)
    
    def update(self, update_actor=True):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        
        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:
                share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch = sample
                old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
                
                adv_targ = adv_targ.to(**self.tpdv)
                value_preds_batch = value_preds_batch.to(**self.tpdv)
                return_batch = return_batch.to(**self.tpdv)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(share_obs_batch,
                                                                                    obs_batch, 
                                                                                    rnn_states_batch, 
                                                                                    rnn_states_critic_batch, 
                                                                                    actions_batch, 
                                                                                    masks_batch, 
                                                                                    available_actions_batch)
                # Actor update
                self.actor_optimizer.zero_grad()
                
                imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = imp_weights * adv_targ
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
                if update_actor:
                    (policy_loss - dist_entropy * self.entropy_coef).backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                
                # Critic Update
                self.critic_optimizer.zero_grad()
                
                value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
                (value_loss * self.value_loss_coef).backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.critic_optimizer.step()
                

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info
        
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
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
    
    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()