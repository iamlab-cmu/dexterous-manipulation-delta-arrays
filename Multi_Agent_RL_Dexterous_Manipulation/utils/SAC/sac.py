from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import utils.SAC.core as core
import torch.nn as nn
import wandb
# from utils.openai_utils.logx import EpochLogger
from utils.SAC.replay_buffer import ReplayBuffer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict(), ma=False, train_or_test="train"):
        # if train_or_test == "train":
        #     self.logger = EpochLogger(**logger_kwargs)
        #     self.logger.save_config(locals())

        self.train_or_test = train_or_test
        # torch.manual_seed(hp_dict['seed'])
        # np.random.seed(hp_dict['seed'])
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['observation_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.batch_size = self.hp_dict['batch_size']

        if ma:
            self.ac = core.MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit, hidden_sizes=[256, 512, 256], activation=nn.GELU)
        else:
            self.ac = core.MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit)
        self.ac_targ = deepcopy(self.ac)

        self.ac = self.ac.to(self.device)
        self.ac_targ = self.ac_targ.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'])

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # if self.train_or_test == "train":
        #     self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=hp_dict['pi_lr'])
        self.q_optimizer = Adam(self.q_params, lr=hp_dict['q_lr'])

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.pi_optimizer, T_0=2, T_mult=2, eta_min=hp_dict['pi_eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.q_optimizer, T_0=2, T_mult=2, eta_min=hp_dict['q_eta_min'])

        self.q_loss = None
        # Set up model saving
        # if self.train_or_test == "train":
        #     self.logger.setup_pytorch_saver(self.ac)

    def compute_q_loss(self, data):
        o, a, r, o2, d = data['obs'].to(self.device), data['act'].to(self.device), data['rew'].to(self.device), data['obs2'].to(self.device), data['done'].to(self.device)
        o = torch.reshape(o, (self.batch_size, -1,))
        a = torch.reshape(a, (self.batch_size, -1,))
        o2 = torch.reshape(o2, (self.batch_size, -1,))
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
        q_loss = loss_q1 + loss_q2

        q_loss.backward()
        ## Uncomment the below if graident is becoming unstable
        # torch.nn.utils.clip_grad_norm_(self.optimizer_critic, self.hp_dict['max_grad_norm'])
        self.q_optimizer.step()

        if not self.hp_dict["dont_log"]:
            self.q_loss = q_loss.cpu().detach().numpy()
            wandb.log({"Q loss":q_loss.cpu().detach().numpy()})

    def compute_pi_loss(self, data):
        for p in self.q_params:
            p.requires_grad = False

        o = data['obs'].to(self.device)
        o = torch.reshape(o, (self.batch_size, -1,))
        # print(o.shape)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        pi_loss = (self.hp_dict['alpha'] * logp_pi - q_pi).mean()
        # Useful info for logging
        
        pi_loss.backward()
        ## Uncomment the below if graident is becoming unstable
        # torch.nn.utils.clip_grad_norm_(self.optimizer_actor, self.hp_dict['max_grad_norm'])
        self.pi_optimizer.step()

        if not self.hp_dict["dont_log"]:
            wandb.log({"Pi loss":pi_loss.cpu().detach().numpy()})
        
        for p in self.q_params:
            p.requires_grad = True

    def update(self, batch_size, current_episode):
        data = self.replay_buffer.sample_batch(batch_size)
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        self.compute_q_loss(data)

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        self.compute_pi_loss(data)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(1 - self.hp_dict['tau'])
                p_targ.data.add_(self.hp_dict['tau'] * p.data)

        if (current_episode % 10000) == 0:
            torch.save(self.ac.state_dict(), f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")

    def get_actions(self, o, deterministic=False):
        obs = torch.tensor(o, dtype=torch.float32).to(self.device)
        obs = torch.reshape(obs, (-1,))
        return self.ac.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)

    def load_saved_policy(self, path):
        self.ac.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        # self.ac.to(device)

    def test_policy(self, o):
        with torch.no_grad():
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32), True) # Generate deterministic policies
        return np.clip(a, -self.act_limit, self.act_limit)
        
        