import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
# from utils.MADP.dit_core import DiffusionTransformer, EMA
from utils.MADP.gpt_adaln_core import Transformer, EMA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MABC:
    def __init__(self):
        self.hp_dict = {
            "exp_name"          : "MATIL_1",
            "data_dir"          : "./data/rl_data",
            'warmup_iters'      : 10,
            'lr'                : 1e-3,
            'eta_min'           : 1e-6,
            "q_eta_min"         : 1e-6,
            "pi_eta_min"        : 1e-6,
            'ckpt_dir'          : './matil_expt_1.pth',
            'idx_embed_loc'     : './utils/MADP/idx_embedding_128.pth',

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            'act_limit'         : 0.03,
            "device"            : torch.device(f"cuda:0"),
            "model_dim"         : 128,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'decoder': 12},
            "dropout"           : 0,
            "max_grad_norm"     : 1,

            "EMA Params":{
                'update_after_step' : 0,
                'inv_gamma'         : 1.0,
                'power'             : 0.75,
                'min_value'         : 0.5,
                'max_value'         : 0.9999,
                }
        }
        self.device = self.hp_dict['device']
        self.model = Transformer(self.hp_dict)
        self.model.to(self.device)
        
        with open('./utils/MADP/normalizer_bc.pkl', 'rb') as f:
            normalizer = pkl.load(f)
        self.obj_name_encoder = normalizer['obj_name_encoder']

    @torch.no_grad()
    def get_actions(self, obs, pos, obj_name, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
        # print(np.array(obj_name).ravel().shape)
        print(obj_name, dict(zip(self.obj_name_encoder.classes_, self.obj_name_encoder.transform(self.obj_name_encoder.classes_))))
        obj_name_enc = torch.as_tensor(self.obj_name_encoder.transform(np.array(obj_name).ravel()), dtype=torch.int32).to(self.device)
            
        # print(obs.size(), pos.size(), obj_name_enc.size())
        # print(obs.dtype, pos.dtype, obj_name_enc.dtype)
        actions, logprob = self.model(obs, obj_name_enc, pos, deterministic=deterministic)
        return actions.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_actions_batch(self, obs, pos, obj_name_enc, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).to(self.device)
        obj_name_enc = obj_name_enc.to(self.device)
            
        # print(obs.size(), pos.size(), obj_name_enc.size())
        # print(obs.dtype, pos.dtype, obj_name_enc.dtype)
        actions, logprob = self.model(obs, obj_name_enc, pos, deterministic=deterministic)
        return actions.detach().cpu().numpy().mean(axis=0)
    # def get_actions(self, states, obj_name_encs, pos, deterministic=False):
    #     bs, n_agents, _ = states.size()
    #     actions = torch.zeros((bs, n_agents, self.hp_dict['action_dim'])).to(self.device)
    #     obj_name_encs = obj_name_encs.long().to(self.hp_dict['device'])
    #     pos = pos[:, :n_agents].to(self.hp_dict['device'])
    #     states = states.to(self.hp_dict['device'])

    #     actions = self.model.get_actions(states, obj_name_encs, pos)
    #     # for i in range(n_agents):
    #     #     updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos)
    #     #     actions = actions.clone()
    #     #     actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
    #     return actions.detach().cpu().numpy()

    def load_saved_policy(self, path):
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(expt_dict['model'])
