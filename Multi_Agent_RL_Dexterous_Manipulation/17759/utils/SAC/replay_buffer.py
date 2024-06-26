import numpy as np
import random
from collections import deque
from copy import deepcopy

# class ReplayBuffer:

#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.buffer = deque(maxlen=max_size)

#     def push(self, state, action, reward, next_state, done):
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []

#         batch = random.sample(self.buffer, batch_size)

#         for experience in batch:
#             state, action, reward, next_state, done = experience
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)
#         return (state_batch, action_batch, np.array(reward_batch), next_state_batch, done_batch)

    
#     def __len__(self):
#         return len(self.buffer)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

