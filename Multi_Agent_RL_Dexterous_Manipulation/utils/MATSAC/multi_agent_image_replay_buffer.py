import numpy as np
import random
from collections import deque
from copy import deepcopy
import torch
import pickle as pkl
import h5py

import utils.DDPG.core as core

class MultiAgentImageReplayBuffer:
    def __init__(self, act_dim, size, max_agents):
        self.act_dim = act_dim
        self.obs_buf = []
        self.obs2_buf = []
        self.act_buf = np.zeros((size, max_agents, act_dim), dtype=np.float32)
        self.pos_buf = np.zeros((size, max_agents, 1), dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.num_agents_buf = np.zeros(size, dtype=np.int32)
        self.obj_names = [None]*size
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, pos, rew, next_obs, done, n_agents, obj_name):
        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
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
        with h5py.File('./data/replay_buffer.h5', 'w') as f:
            # Placeholder for images dataset
            image_shape = self.obs_buf[0].shape
            num_images = len(self.obs_buf)
            
            # Create a dataset to store images, adjust dtype to match your image data type
            img_dset = f.create_dataset('images', shape=(num_images, *image_shape), dtype='uint8')
            action_dset = f.create_dataset('actions', shape=(num_images, *self.act_dim), dtype='float32')
            reward_dset = f.create_dataset('reward', shape=(num_images, 1), dtype='float32')
            obj_names_dset = f.create_dataset('obj_names', shape=(num_images,), dtype=h5py.string_dtype(encoding='utf-8'))
            pos_dset = f.create_dataset('pos', shape=(num_images, 1), dtype='int64')
            num_agents_dset = f.create_dataset('num_agents', shape=(num_images, 1), dtype='int32')

            # Iterate over the data and save to HDF5
            for i in range(num_images):
                img_dset[i] = self.obs_buf[i]  # Store the image
                action_dset[i] = self.act_buf[i]  # Store the action
                reward_dset[i] = self.rew_buf[i]
                obj_names_dset[i] = self.obj_names[i]
                pos_dset[i] = self.pos_buf[i]
                num_agents_dset[i] = self.num_agents_buf[i]


        # dic = { "obs":self.obs_buf,
        #         "obs2":self.obs2_buf,
        #         "act":self.act_buf,
        #         "pos":self.pos_buf,
        #         "rew":self.rew_buf,
        #         "done":self.done_buf,
        #         "num_agents":self.num_agents_buf,
        #         'obj_names':self.obj_names}
        # pkl.dump(dic, open("replay_buffer.pkl", "wb"))



        # np.savez("replay_buffer.npz", obs_buf=self.obs_buf, obs2_buf=self.obs2_buf, act_buf=self.act_buf, rew_buf=self.rew_buf, done_buf=self.done_buf, num_agents_buf=self.num_agents_buf)
