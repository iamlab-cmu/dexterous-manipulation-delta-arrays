import numpy as np
import torch
import pickle as pkl

class MultiAgentReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, max_agents):
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
        for (obs, act, _, _, pos, rew, next_obs, done, n_agents) in replay_data:
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


class MultiAgentImageReplayBuffer:
    def __init__(self, act_dim, size, max_agents):
        self.act_dim = act_dim
        self.max_agents = max_agents
        self.obs_buf = []
        self.obs2_buf = []
        self.goal = []
        self.act_buf = np.zeros((size, max_agents, act_dim), dtype=np.float32)
        self.pos_buf = np.zeros((size, max_agents, 1), dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.num_agents_buf = np.zeros(size, dtype=np.int32)
        self.obj_names = [None]*size
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, next_obs, goal, act, pos, rew, done, n_agents, obj_name):
        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
        self.goal.append(goal)
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
        batch = dict(obs1=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     goal=self.goal[idxs],
                     act=self.act_buf[idxs],
                     pos=self.pos_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     num_agents=self.num_agents_buf[idxs],
                     obj_names=self.obj_names,)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def save_RB(self):
        dic = { "obs1":self.obs_buf,
                "obs2":self.obs2_buf,
                "goal":self.goal,
                "act":self.act_buf,
                "pos":self.pos_buf,
                "rew":self.rew_buf,
                "done":self.done_buf,
                "num_agents":self.num_agents_buf,
                'obj_names':self.obj_names}
        pkl.dump(dic, open("replay_buffer.pkl", "wb"))
        

class MADiffTD3ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, max_agents, diff_steps):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, max_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, max_agents, act_dim), dtype=np.float32)
        self.pos_buf = np.zeros((size, max_agents), dtype=np.int32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.num_agents_buf = np.zeros(size, dtype=np.int32)
        self.low_level_mdp_logps = np.zeros((size, diff_steps, max_agents), dtype=np.float32)
        self.low_level_mdp_a_ks = np.zeros((size, diff_steps, max_agents, act_dim), dtype=np.float32)
        
        self.ptr, self.sample_ptr,  self.size, self.max_size = 0, 0, 0, size

    def store(self, replay_data):
        for (obs, act, log_ps, a_ks, pos, rew, next_obs, done, n_agents) in replay_data:
            self.obs_buf[self.ptr, :n_agents] = obs
            self.obs2_buf[self.ptr, :n_agents] = next_obs
            self.act_buf[self.ptr, :n_agents] = act
            self.pos_buf[self.ptr, :n_agents] = pos
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.num_agents_buf[self.ptr] = n_agents
            self.low_level_mdp_logps[self.ptr, :, :n_agents] = log_ps
            self.low_level_mdp_a_ks[self.ptr, :, :n_agents] = a_ks #: represents 100 denoising steps (k)
            
            self.ptr = (self.ptr+1) % self.max_size
            if self.ptr == 0:
                self.sample_ptr = 0
            self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        ppo_idxs = np.random.randint(self.sample_ptr, self.ptr, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    pos=self.pos_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    num_agents=self.num_agents_buf[idxs],
                    log_ps=self.low_level_mdp_logps[ppo_idxs],
                    a_ks=self.low_level_mdp_a_ks[ppo_idxs],
                )
        
        return {k: torch.as_tensor(v) for k,v in batch.items()}
    
    def update_sample_ptr(self):
        self.sample_ptr = self.ptr

    def save_RB(self):
        dic = { "obs":self.obs_buf,
                "obs2":self.obs2_buf,
                "act":self.act_buf,
                "pos":self.pos_buf,
                "rew":self.rew_buf,
                "done":self.done_buf,
                "num_agents":self.num_agents_buf,
                "log_ps":self.low_level_mdp_logps,
                "a_ks":self.low_level_mdp_a_ks,}
        pkl.dump(dic, open("./data/replay_buffer_mixed.pkl", "wb"))