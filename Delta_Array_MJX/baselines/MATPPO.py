import numpy as np
from copy import deepcopy
import itertools
import wandb
import scipy.signal

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import baselines.gpt_ppo as core
import baselines.multi_agent_ppo_replay_buffer as MARB
from baselines.utils import ValueNorm, huber_loss

class MATPPO:
    def __init__(self, env_dict, hp_dict, train_or_test="train"):
        self.train_or_test = train_or_test
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.gauss = hp_dict['gauss']
        self.log_dict = {
            'V loss': [],
            'Pi loss': [],
            'KL Div': [],
            'adv': [],
            'entropy': [],
            'Reward': []
        }
        self.max_avg_rew = 0
        self.batch_size = hp_dict['batch_size']
        self.epsilon = hp_dict['ppo_clip']
        self.H_coeff = hp_dict['H_coef']
        self.gamma = hp_dict['gamma']
        self.gae_lambda = hp_dict['gae_lambda']

        self.tf = core.Transformer(self.hp_dict).to(self.device)
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(hp_dict['nenv'], self.obs_dim, self.act_dim, hp_dict['rblen'], self.tf.max_agents)

        var_counts = tuple(core.count_vars(module) for module in [self.tf])
        if self.train_or_test == "train":
            print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        if self.hp_dict['optim']=="adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.tf.parameters()), lr=hp_dict['pi_lr'])
        elif self.hp_dict['optim']=="adamW":
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.parameters()), lr=hp_dict['pi_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.tf.parameters()), lr=hp_dict['pi_lr'])

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=hp_dict['eta_min'])
        
        self.internal_updates_counter = 0
        
        self.pi_loss_scaler = torch.amp.GradScaler('cuda')
        
        self.value_normalizer = ValueNorm(1, device=self.device)

    def compute_v_loss(self, val_gt, val, returns):
        val_gt_clip = val_gt + (val - val_gt).clamp(-self.epsilon, self.epsilon)
        self.value_normalizer.update(returns)
        err_clipped = self.value_normalizer.normalize(returns) - val_gt_clip
        err_unclipped = self.value_normalizer.normalize(returns) - val
        val_loss_clipped = huber_loss(err_clipped, 10.0)
        val_loss_original = huber_loss(err_unclipped, 10.0)
        v_loss = torch.max(val_loss_original, val_loss_clipped).mean()
        return v_loss

    def compute_pi_loss(self, s1, val_gt, a, logp, adv_tgt, returns, pos):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            val, log_probs, H = self.tf(s1, a, pos)
            val = val.view(-1, 1)
            
            log_probs = log_probs.reshape(-1, self.act_dim)
            imp_weights = torch.exp(log_probs - logp)
            
            surr1 = imp_weights * adv_tgt
            surr2 = torch.clamp(imp_weights, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_tgt
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            
            v_loss = self.compute_v_loss(val_gt, val, returns)
            loss = policy_loss - self.H_coeff * H + v_loss
            
        self.pi_loss_scaler.scale(loss).backward()
        self.pi_loss_scaler.unscale_(self.optimizer)
        # pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.parameters(), self.hp_dict['max_grad_norm'])
        self.pi_loss_scaler.step(self.optimizer)
        self.pi_loss_scaler.update()
        
        # Log Values for Wandb
        self.log_dict['V loss'].append(v_loss.item())
        self.log_dict['Pi loss'].append(policy_loss.item())
        self.log_dict['entropy'].append(H.mean().item())
        self.log_dict['adv'].append(adv_tgt.mean().item())
        self.log_dict['KL Div'].append((logp - log_probs).mean().item())

    def discount_cumsum(self, x, discount):
        """ magic from rllab for computing discounted cumulative sums of vectors. """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def compute_returns(self, rew, val_gt):
        val_t_next = self.value_normalizer.denormalize(val_gt[1:])
        val_t = self.value_normalizer.denormalize(val_gt[:-1])
        
        deltas = rew + self.gamma * val_t_next - val_t
        adv = self.discount_cumsum(deltas, self.gamma * self.gae_lambda)
        returns = self.discount_cumsum(rew, self.gamma)
        
        adv = torch.as_tensor(adv.copy(), dtype=torch.float32).view(-1, 1).to(self.device)
        returns = torch.as_tensor(returns.copy(), dtype=torch.float32).view(-1, 1).to(self.device)
        val_gts = torch.as_tensor(val_gt[:-1], dtype=torch.float32).reshape(-1, 1).to(self.device)
        return adv, returns, val_gts
        # gae = 0
        # for i in reversed(range(rew.shape[0])):
        #     rew_t = rew[i]
        #     mean_v_t = torch.mean(val[i], axis=-2, keepdim=True)
        #     mean_v_t_next = torch.mean(val[i+1], axis=-2, keepdim=True)
        #     delta = rew + self.gamma * mean_v_t_next - mean_v_t
            
        #     gae = 

    def update(self, env_id, current_episode, reward, n_updates, good_terminate=False):
        self.log_dict['Reward'].append(reward)
        for j in range(n_updates):
            self.internal_updates_counter += 1
            self.scheduler.step()

            data = self.ma_replay_buffer.sample_rb(env_id)
            n_agents = int(torch.max(data['num_agents']))
            states = data['obs'][:,:n_agents].to(self.device)
            actions = data['act'][:,:n_agents].to(self.device)
            logp = data['logp'][:,:n_agents].reshape(-1, self.act_dim).to(self.device)
            val_gts = data['val'][:,:n_agents]
            rews = data['rew'][:,:n_agents]
            if good_terminate:
                with torch.no_grad():
                    val_gts[-1] = self.tf.get_val(data['obs2'][-1,:n_agents].to(self.device), data['pos'][-1,:n_agents].to(self.device)).squeeze().cpu().numpy()
            adv, returns, val_gts = self.compute_returns(rews, val_gts)
            
            pos = data['pos'][:,:n_agents].to(self.device)
            # self.log_dict['Reward'].append(rews.mean())
            
            for param in self.tf.decoder_actor.parameters():
                param.grad = None
            self.compute_pi_loss(states, val_gts, actions, logp, adv, returns, pos)
                        
            if (self.train_or_test == "train") and (self.internal_updates_counter % 5000 == 0):
                    print("ckpt saved @ ", current_episode, self.internal_updates_counter)
                    dicc = {
                        'model': self.tf.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(dicc, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
        
        if (not self.hp_dict["dont_log"]) and (self.internal_updates_counter % 50 == 0):
            wandb.log({k: np.mean(v) if isinstance(v, list) and len(v) > 0 else v for k, v in self.log_dict.items()})
            self.log_dict = {
                'V loss': [],
                'Pi loss': [],
                'adv': [],
                'entropy': [],
                'Reward': [],
                'KL Div': []
            }
        
        self.ma_replay_buffer.reset_rb(env_id)
                
    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        
        actions, logp_pi, val = self.tf.get_actions(obs, pos, deterministic=deterministic)
        if not deterministic:
            return actions.detach().cpu().numpy()[0], logp_pi.detach().cpu().numpy(), val.detach().cpu().numpy()
        else:
            return actions.detach().cpu().numpy()[0], None, val.detach().cpu().numpy()
    
    def save_policy(self):
        dicc = {
            'model': self.tf.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(dicc, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        dicc = torch.load(path, map_location=self.hp_dict['dev_rl'])
        
        self.tf.load_state_dict(dicc['model'])
        self.optimizer.load_state_dict(dicc['optimizer'])