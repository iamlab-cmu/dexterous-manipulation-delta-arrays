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

class MABC:
    def __init__(self):
        self.hp_dict = {
        "exp_name"          : "MATIL_1",
        "data_dir"          : "./data/rl_data",
        'warmup_iters'      : 10,
        'lr'                : 1e-3,
        'eta_min'           : 1e-6,
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
        self.model.to(self.hp_dict['device'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hp_dict['lr'], weight_decay=0)

    def get_actions(self, states, obj_name_encs, pos, deterministic=False):
        bs, n_agents, _ = states.size()
        actions = torch.zeros((bs, n_agents, self.hp_dict['action_dim'])).to(self.device)
        obj_name_encs = obj_name_encs.long().to(self.hp_dict['device'])
        pos = pos[:, :n_agents].to(self.hp_dict['device'])
        states = states.to(self.hp_dict['device'])

        actions = self.model.get_actions(states, obj_name_encs, pos)
        # for i in range(n_agents):
        #     updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos)

        #     # TODO: Ablate here with all actions cloned so that even previous actions are updated with new info. 
        #     # TODO: Does it cause instability? How to know if it does?
        #     actions = actions.clone()
        #     actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
        return actions.detach().cpu().numpy()

    def load_saved_policy(self, path):
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(expt_dict['model'])
