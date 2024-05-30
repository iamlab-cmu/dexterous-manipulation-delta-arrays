import numpy as np
import random
from collections import deque
from copy import deepcopy
import torch
import pickle as pkl

import utils.DDPG.core as core

class MultiAgentReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, max_agents):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, max_agents, act_dim), dtype=np.float32)
        self.pos_buf = np.zeros((size, max_agents, 1), dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.num_agents_buf = np.zeros(size, dtype=np.int32)
        self.obj_names = [None]*size
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, pos, rew, next_obs, done, n_agents, obj_name):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.pos_buf[self.ptr] = pos
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.num_agents_buf[self.ptr] = n_agents
        self.obj_names[self.ptr] = obj_name
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     pos=self.pos_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     num_agents=self.num_agents_buf[idxs],
                    )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def save_RB(self):
        dic = { "obs":self.obs_buf,
                "obs2":self.obs2_buf,
                "act":self.act_buf,
                "pos":self.pos_buf,
                "rew":self.rew_buf,
                "done":self.done_buf,
                "num_agents":self.num_agents_buf,
                'obj_names':self.obj_names}
        pkl.dump(dic, open("replay_buffer.pkl", "wb"))
        # np.savez("replay_buffer.npz", obs_buf=self.obs_buf, obs2_buf=self.obs2_buf, act_buf=self.act_buf, rew_buf=self.rew_buf, done_buf=self.done_buf, num_agents_buf=self.num_agents_buf)
