import numpy as np
import torch
import pickle as pkl

class MultiAgentReplayBuffer:
    def __init__(self, n_threads, obs_dim, act_dim, size, max_agents):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, max_agents, act_dim), dtype=np.float32)
        self.pos_buf = np.zeros((size, max_agents), dtype=np.int32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.num_agents_buf = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, replay_data):
        for (obs, act, pos, rew, next_obs, done, n_agents) in replay_data:
            self.obs_buf[self.ptr, :n_agents] = obs
            self.obs2_buf[self.ptr, :n_agents] = next_obs
            self.act_buf[self.ptr, :n_agents] = act
            self.pos_buf[self.ptr, :n_agents] = pos
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.num_agents_buf[self.ptr] = n_agents
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
                     num_agents=self.num_agents_buf[idxs],)
        
        return {k: torch.as_tensor(v) for k,v in batch.items()}

    def save_RB(self):
        dic = { "obs":self.obs_buf,
                "obs2":self.obs2_buf,
                "act":self.act_buf,
                "pos":self.pos_buf,
                "rew":self.rew_buf,
                "done":self.done_buf,
                "num_agents":self.num_agents_buf,}
        pkl.dump(dic, open("./data/replay_buffer_mixed.pkl", "wb"))