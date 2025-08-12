import os
import numpy as np
import pickle as pkl
import time

from utils.constants import RB_STORE, RB_SAMPLE, RB_SAVE, RB_LOAD, RB_CLOSE

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size, max_agents):
        self.ptr, self.size, self.max_size = 0, 0, max_size

        self.st_buf         = np.zeros((max_size, max_agents, obs_dim), dtype=np.float32)
        self.pos_buf        = np.zeros((max_size, max_agents), dtype=np.int32)
        self.at_buf         = np.zeros((max_size, max_agents, act_dim), dtype=np.float32)
        self.rew_buf        = np.zeros((max_size,), dtype=np.float32)
        self.st_plus_1_buf  = np.zeros((max_size, max_agents, obs_dim), dtype=np.float32)
        self.done_buf       = np.zeros((max_size,), dtype=np.float32)

    def store(self, batch_of_transitions):
        obs, action, pos, reward, next_obs, done = batch_of_transitions
        batch_size = obs.shape[0]
        
        if self.ptr + batch_size > self.max_size:
            part1_size = self.max_size - self.ptr
            part2_size = batch_size - part1_size
            
            self.st_buf[self.ptr:] = obs[:part1_size]
            self.at_buf[self.ptr:] = action[:part1_size]
            self.rew_buf[self.ptr:] = reward[:part1_size]
            self.st_plus_1_buf[self.ptr:] = next_obs[:part1_size]
            self.done_buf[self.ptr:] = done[:part1_size]
            self.pos_buf[self.ptr:] = pos[:part1_size]
            
            self.st_buf[:part2_size] = obs[part1_size:]
            self.at_buf[:part2_size] = action[part1_size:]
            self.rew_buf[:part2_size] = reward[part1_size:]
            self.st_plus_1_buf[:part2_size] = next_obs[part1_size:]
            self.done_buf[:part2_size] = done[part1_size:]
            self.pos_buf[:part2_size] = pos[part1_size:]
            
        else:
            idxs = np.arange(self.ptr, self.ptr + batch_size)
            self.st_buf[idxs] = obs
            self.at_buf[idxs] = action
            self.rew_buf[idxs] = reward
            self.st_plus_1_buf[idxs] = next_obs
            self.done_buf[idxs] = done
            self.pos_buf[idxs] = pos

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = (
            self.st_buf[idxs],
            self.at_buf[idxs],
            self.rew_buf[idxs],
            self.st_plus_1_buf[idxs],
            self.done_buf[idxs],
            self.pos_buf[idxs]
        )
        return batch
    
    def save_RB(self):
        print("Dumping replay buffer")
        with open(os.path.join(self.config['path'], "replay_buffer.pkl"), "wb") as f:
            dic = { "st_buf": self.st_buf,
                    "at_buf": self.at_buf,
                    "rew_buf": self.rew_buf,
                    "st_plus_1_buf": self.st_plus_1_buf,
                    "done_buf": self.done_buf,
                    "pos_buf": self.pos_buf, }
            pkl.dump(dic, f)

    def load_RB(self):
        print("Loading replay buffer")
        with open(os.path.join(self.config['path'], "replay_buffer.pkl"), "rb") as f:
            dic = pkl.load(f)
            self.st_buf         = dic["st_buf"]
            self.at_buf         = dic["at_buf"]
            self.rew_buf        = dic["rew_buf"]
            self.st_plus_1_buf  = dic["st_plus_1_buf"]
            self.done_buf       = dic["done_buf"]
            self.pos_buf        = dic["pos_buf"]
    
    def __len__(self):
        return self.size
            
def rb_worker_continuous(config, rb_queue, rb_response):
    batch_size = 0
    replay_buffer = ReplayBuffer(config)
    
    while True:
        if not (rb_queue.empty()):
            req_code, data = rb_queue.get()

            if req_code == RB_STORE:
                replay_buffer.store(data)
                
            elif req_code == RB_SAMPLE:
                batch_size = data
                batch = replay_buffer.sample_batch(batch_size)
                rb_response.put(batch)
                
            elif req_code == RB_SAVE:
                replay_buffer.save_RB()
                
            elif req_code == RB_LOAD:
                replay_buffer.load_RB()
                
            elif req_code == RB_CLOSE:
                print("Closing replay buffer worker")
                break
            
        elif batch_size > 0 and (not rb_response.full()):
            batch = replay_buffer.sample_batch(batch_size)
            rb_response.put(batch)
            
        time.sleep(0.0005)