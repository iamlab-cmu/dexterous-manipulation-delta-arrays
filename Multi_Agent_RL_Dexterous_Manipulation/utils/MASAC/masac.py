from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import utils.MASAC.core as core
from utils.openai_utils.logx import EpochLogger
from utils.MASAC.multi_agent_replay_buffer import MultiAgentReplayBuffer
import itertools

class MASAC:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), train_or_test="train"):
        if train_or_test == "train":
            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(locals())

        self.train_or_test = train_or_test
        # torch.manual_seed(hp_dict['seed'])
        # np.random.seed(hp_dict['seed'])
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.pi_obs_dim = self.env_dict['pi_obs_dim']['dim']
        self.q_obs_dim = self.env_dict['pi_obs_dim']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.max_agents = self.env_dict['max_agents']

        self.act_limit = self.env_dict['action_space']['high']
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))

        self.ac = core.MLPActorCritic(self.pi_obs_dim, self.q_obs_dim, self.act_dim, self.act_dim*self.max_agents, self.act_limit)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.ma_replay_buffer = MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'])

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if self.train_or_test == "train":
            self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=hp_dict['pi_lr'])
        self.q_optimizer = Adam(self.q_params, lr=hp_dict['q_lr'])

        # Set up model saving
        if self.train_or_test == "train":
            self.logger.setup_pytorch_saver(self.ac)

    def compute_q_loss(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q function. For 1 step quasi static policy, this is just reward value. No discounted returns.``
        with torch.no_grad():
            # q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            # backup = r + gamma * (1 - d) * q_pi_targ
            a2, logp_a2 = self.ac.pi(o2)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.hp_dict['gamma'] * (1 - d) * (q_pi_targ - self.hp_dict['alpha'] * logp_a2)
        
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(),
                        Q2Vals=q2.detach().numpy())
        return loss_q, q_info

    def compute_pi_loss(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        loss_pi = (self.hp_dict['alpha'] * logp_pi - q_pi).mean()
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())
        return loss_pi, pi_info

    def update(self, batch_size, n_agents):
        data = self.ma_replay_buffer.sample_batch(batch_size)
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_q_loss(data, n_agents)
        loss_q.backward()
        self.q_optimizer.step()
        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_pi_loss(data, n_agents)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(1 - self.hp_dict['tau'])
                p_targ.data.add_(self.hp_dict['tau'] * p.data)

    def get_actions(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def load_saved_policy(self, path='./data/rl_data/backup/sac_expt_grasp/pyt_save/model.pt'):
        self.ac.load_state_dict(torch.load(path))

    def test_policy(self, o):
        with torch.no_grad():
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32), True) # Generate deterministic policies
        return np.clip(a, -self.act_limit, self.act_limit)
        
        