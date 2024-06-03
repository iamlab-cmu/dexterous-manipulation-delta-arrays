import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from einops import rearrange, reduce

import pytorch_warmup as warmup

from dit_core import DiffusionTransformer, EMA, _extract_into_tensor

rb_pos_world = np.zeros((8,8,2))
kdtree_positions_world = np.zeros((64, 2))
for i in range(8):
    for j in range(8):
        if i%2!=0:
            finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
        else:
            finger_pos = np.array((i*0.0375, j*0.043301))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
        kdtree_positions_world[i*8 + j, :] = rb_pos_world[i,j]

class DataNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        reshaped_data = data.reshape(-1, data.shape[-1])
        self.scaler.fit(reshaped_data)
        return self

    def transform(self, data):
        reshaped_data = data.reshape(-1, data.shape[-1])
        transformed_data = self.scaler.transform(reshaped_data)
        return transformed_data.reshape(data.shape)

    def inverse_transform(self, data):
        reshaped_data = data.reshape(-1, data.shape[-1])
        inverse_transformed_data = self.scaler.inverse_transform(reshaped_data)
        return inverse_transformed_data.reshape(data.shape)

class ImitationDataset(Dataset):
    def __init__(self, states, actions, state_scaler, action_scaler, pos, num_agents, obj_names, obj_of_interest=None):
        self.states = states
        self.actions = actions
        self.pos = pos
        self.num_agents = num_agents
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

        obj_names = np.array(obj_names)
        names = ['block', 'disc', 'hexagon', 'parallelogram', 'semicircle', 'shuriken', 'star', 'trapezium', 'triangle']
        # encoder = OneHotEncoder(sparse=False)
        encoder = LabelEncoder()
        # self.obj_names_encoded = encoder.fit_transform(np.array(obj_names).reshape(-1, 1))
        self.obj_names_encoded = encoder.fit_transform(np.array(obj_names).ravel())

        if obj_of_interest:
            idxs = np.where(obj_names == obj_of_interest)[0]
            self.states = self.states[idxs]
            self.actions = self.actions[idxs]
            self.pos = self.pos[idxs]
            self.num_agents = self.num_agents[idxs]
            self.obj_names_encoded = self.obj_names_encoded[idxs]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        obj_names_encoded = self.obj_names_encoded[idx]
        pos = self.pos[idx]
        num_agents = self.num_agents[idx]
        return state, action, obj_names_encoded, pos, num_agents

def get_strings_by_indices(strings_list, indices_array):
    result = []
    for index in indices_array:
        result.append(strings_list[index])
    return result

def get_smol_dataset(states, actions, state_scaler, action_scaler, pos, num_agents, obj_names, num_samples:int=None, obj_of_interest: str=None):
    if num_samples is None:
        return ImitationDataset(states, actions, state_scaler, action_scaler, pos, num_agents, obj_names, obj_of_interest=None)
        
    unique_object_names = np.unique(obj_names)
    final_indices = []
    for obj_name in unique_object_names:
        obj_indices = np.where(obj_names == obj_name)[0]
        chosen_indices = np.random.choice(obj_indices, num_samples, replace=False)
        final_indices.extend(chosen_indices)

    final_indices = np.array(final_indices)

    smol_states = states[final_indices]
    smol_actions = actions[final_indices]
    smol_pos = pos[final_indices]
    smol_obj_names = obj_names[final_indices]
    smol_num_agents = num_agents[final_indices]
    return ImitationDataset(smol_states, smol_actions, state_scaler, action_scaler, smol_pos, smol_num_agents, smol_obj_names, obj_of_interest=obj_of_interest)

def get_dataset_and_dataloaders(data_pkg, train_bs:int=128, test_bs:int=1, num_samples:int=1000, obj_of_interest=None, rb_path='../../data/replay_buffer.pkl'):
    states, actions, pos, num_agents, obj_names, state_scaler, action_scaler = data_pkg

    dataset = get_smol_dataset(states, actions, state_scaler, action_scaler, pos, num_agents, obj_names, num_samples, obj_of_interest)

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    train_indices, temp_indices = train_test_split(list(range(len(dataset))), test_size=(val_size + test_size))
    val_indices, test_indices = train_test_split(temp_indices, test_size=(test_size / (test_size + val_size)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
    return train_loader, val_loader, test_loader, state_scaler, action_scaler

def train(data_pkg, hp_dict):
    train_loader, val_loader, test_loader, state_scaler, action_scaler = get_dataset_and_dataloaders(data_pkg, train_bs=128, test_bs=128, num_samples=hp_dict['num_samples'], obj_of_interest=None)
    model = DiffusionTransformer(hp_dict)
    ema_model = deepcopy(model).to(hp_dict['device'])
    model.to(hp_dict['device'])
    optimizer = optim.AdamW(model.parameters(), lr=hp_dict['lr'], weight_decay=1e-6)

    # optimizer = optim.SGD(model.parameters(), lr=1e-2)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=hp_dict['eta_min'])

    ema = EMA(ema_model, **hp_dict['EMA Params'])

    losses = []
    global_Step = 0
    start_value = 1
    end_value = hp_dict['lr']/hp_dict['eta_min']
    steps = hp_dict['warmup_iters']

    increment = (end_value - start_value) / (steps - 1)
    values = [start_value + i * increment for i in range(steps)]

    for param_group in optimizer.param_groups:
        param_group['lr'] = hp_dict['eta_min']

    for epoch in range(hp_dict['epochs']):
        with tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False) as t:
            for i, (states, actions, obj_name_encs, pos, num_agents) in enumerate(t):
                n_agents = int(torch.max(num_agents))
                
                # Noised and Denoised Variable
                actions = actions[:, :n_agents].to(hp_dict['device'])

                # Conditioning Variables
                states = states[:, :n_agents].to(hp_dict['device'])
                obj_name_encs = obj_name_encs.long().to(hp_dict['device'])
                pos = pos[:, :n_agents].to(hp_dict['device'])

                optimizer.zero_grad()
                loss = model.compute_loss(actions, states, obj_name_encs, pos)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp_dict['max_grad_norm'])
                optimizer.step()
                ema.step(model)

                if global_Step >= (steps-1):
                    lr_scheduler.step()
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = hp_dict['eta_min'] * values[global_Step]

                losses.append(loss.item())
                t.set_postfix(loss=np.mean(losses[-300:]), refresh=False)
                global_Step += 1
            
            ema_model.eval()
    
    val_losses = []
    with torch.no_grad():
        with tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=True) as t:
            for i, (states, actions, obj_name_encs, pos, num_agents) in enumerate(t):
                n_agents = int(torch.max(num_agents))
                
                actions = actions[:, :n_agents].to(hp_dict['device'])

                states = states[:, :n_agents].to(hp_dict['device'])
                obj_name_encs = obj_name_encs.long().to(hp_dict['device'])
                pos = pos[:, :n_agents].to(hp_dict['device'])

                noise = torch.randn((1, n_agents, 2), device=hp_dict['device'])

                optimizer.zero_grad()
                denoised_actions = model.actions_from_denoising_diffusion(noise, states, obj_name_encs, pos)
                

                val_losses.append(F.mse_loss(actions, denoised_actions).item())

                if i==5:
                    break

    return np.mean(val_losses)