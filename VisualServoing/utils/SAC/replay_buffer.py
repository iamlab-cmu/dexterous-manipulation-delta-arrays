import numpy as np
import random
from collections import deque


class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    
    def __len__(self):
        return len(self.buffer)

# class ReplayBuffer:
#     def __init__(self, max_size):
#         self.buffer = deque(maxlen=max_size)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, np.array([reward]), next_state, done))

#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), np.concatenate(action), np.concatenate(reward), np.concatenate(next_state), done
    
#     def sample_sequence(self, batch_size, seq_len):
#         idx = np.random.randint(0, len(self.buffer) - seq_len, batch_size)
#         state, action, reward, next_state, done = zip(*[self.buffer[i:i+seq_len] for i in idx])
#         return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
#     def __len__(self):
#         return len(self.buffer)