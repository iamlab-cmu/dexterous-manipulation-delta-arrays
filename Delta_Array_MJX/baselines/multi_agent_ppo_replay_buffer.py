import torch
import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, n_threads, obs_dim, act_dim, size, max_agents):
        """
        Nominally, nthreads = 40, max_size = 50, max_agents = 64, obs_dim = 6, act_dim = 2
        """
        self.obs_dim = obs_dim
        self.max_agents = max_agents
        self.max_size = size
        self.act_dim = act_dim
        self.n_threads = n_threads
        self.obss = np.zeros((self.n_threads, self.max_size, self.max_agents, self.obs_dim), dtype=np.float32)
        self.obs2s = np.zeros((self.n_threads, self.max_size, self.max_agents, self.obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.n_threads, self.max_size, self.max_agents, self.act_dim), dtype=np.float32)
        self.act_log_probss = np.zeros((self.n_threads, self.max_size, self.max_agents, self.act_dim), dtype=np.float32)
        self.vals = np.zeros((self.n_threads, self.max_size+1, self.max_agents), dtype=np.float32)
        self.rews = np.zeros((self.n_threads, self.max_size, self.max_agents), dtype=np.float32)
        self.poss = np.zeros((self.n_threads, self.max_size, self.max_agents), dtype=np.int32)
        self.dones = np.zeros((self.n_threads, self.max_size), dtype=np.float32)
        self.num_agentss = np.zeros((self.n_threads, self.max_size), dtype=np.int32)
        self.ptr = np.zeros(self.n_threads, dtype=np.int32)

    def store(self, replay_data):
        env_id, rd = replay_data
        for (obs, act, act_log_prob, val, pos, rew, next_obs, done, n_agents) in rd:
            self.obss[env_id, self.ptr[env_id], :n_agents] = obs
            self.obs2s[env_id, self.ptr[env_id], :n_agents] = next_obs
            self.acts[env_id, self.ptr[env_id], :n_agents] = act
            self.act_log_probss[env_id, self.ptr[env_id], :n_agents] = act_log_prob
            self.vals[env_id, self.ptr[env_id], :n_agents] = val
            self.poss[env_id, self.ptr[env_id], :n_agents] = pos
            self.rews[env_id, self.ptr[env_id], :n_agents] = rew
            self.dones[env_id, self.ptr[env_id]] = done
            self.num_agentss[env_id, self.ptr[env_id]] = n_agents
            self.ptr[env_id] += 1
        assert self.ptr[env_id] <= self.max_size

    def sample_rb(self, env_id):
        batch = dict(obs=self.obss[env_id],
                    obs2=self.obs2s[env_id],
                    act=self.acts[env_id],
                    logp=self.act_log_probss[env_id],
                    pos=self.poss[env_id],
                    done=self.dones[env_id],
                    num_agents=self.num_agentss[env_id],)
        batch2 = {k: torch.as_tensor(v) for k,v in batch.items()}
        batch2["val"] = self.vals[env_id]
        batch2["rew"] = self.rews[env_id]
        return batch2

    def reset_rb(self, env_id):
        self.obss[env_id] = np.zeros((self.max_size, self.max_agents, self.obs_dim), dtype=np.float32)
        self.obs2s[env_id] = np.zeros((self.max_size, self.max_agents, self.obs_dim), dtype=np.float32)
        self.acts[env_id] = np.zeros((self.max_size, self.max_agents, self.act_dim), dtype=np.float32)
        self.act_log_probss[env_id] = np.zeros((self.max_size, self.max_agents, self.act_dim), dtype=np.float32)
        self.vals[env_id] = np.zeros((self.max_size+1, self.max_agents), dtype=np.float32)
        self.rews[env_id] = np.zeros((self.max_size, self.max_agents), dtype=np.float32)
        self.poss[env_id] = np.zeros((self.max_size, self.max_agents), dtype=np.int32)
        self.dones[env_id] = np.zeros(self.max_size, dtype=np.float32)
        self.num_agentss[env_id] = np.zeros(self.max_size, dtype=np.int32)
        self.ptr[env_id] = 0