from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import utils.DDPG.core as core
from utils.openai_utils.logx import EpochLogger
from utils.DDPG.replay_buffer import ReplayBuffer

class DDPG:
    def __init__(self, env_dict, hp_dict, logger_kwargs=dict()):
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # torch.manual_seed(hp_dict['seed'])
        # np.random.seed(hp_dict['seed'])
        self.hp_dict = hp_dict
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['observation_space']['dim']
        self.act_dim = self.env_dict['action_space']['dim']

        self.act_limit = self.env_dict['action_space']['high']

        self.ac = core.MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['replay_size'])

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)
        
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=hp_dict['pi_lr'])
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=hp_dict['q_lr'])

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

    def compute_q_loss(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(o,a)

        # Bellman backup for Q function. For 1 step quasi static policy, this is just reward value. No discounted returns.``
        with torch.no_grad():
            # q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            # backup = r + gamma * (1 - d) * q_pi_targ
            backup = r

        loss_q = ((q - backup)**2).mean()
        loss_info = dict(QVals=q.detach().numpy())
        return loss_q, loss_info

    def compute_pi_loss(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, batch_size):
        data = self.replay_buffer.sample_batch(batch_size)
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_q_loss(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_pi_loss(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(1 - self.hp_dict['tau'])
                p_targ.data.add_(self.hp_dict['tau'] * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def load_saved_policy(self, path):
        self.ac.load_state_dict(torch.load('./data/rl_data/ddpg_expt_0/ddpg_expt_0_s69420/pyt_save/model.pt'))

    def test_policy(self, o):
        with torch.no_grad():
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        return np.clip(a, -self.act_limit, self.act_limit)
        
        