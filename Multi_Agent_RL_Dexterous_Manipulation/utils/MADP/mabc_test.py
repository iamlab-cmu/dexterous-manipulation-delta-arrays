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
from utils.MADP.gpt_adaln_no_autoreg import Transformer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MABC:
    def __init__(self, parent_hp_dict):
        self.hp_dict = {
            "exp_name"          : parent_hp_dict['exp_name'],
            "data_dir"          : "./data/rl_data",
            "ckpt_loc"          : "./utils/MADP/mabc_new_data_ac_gauss.pth",
            "dont_log"          : parent_hp_dict['dont_log'],
            "replay_size"       : 500001,
            'warmup_epochs'     : 1000,
            'pi_lr'             : parent_hp_dict['pi_lr'],
            'q_lr'              : parent_hp_dict['q_lr'],
            "q_eta_min"         : parent_hp_dict['q_eta_min'],
            "pi_eta_min"        : parent_hp_dict['pi_eta_min'],
            'ckpt_dir'          : './mabc_finetune_final.pth',
            'idx_embed_loc'     : './utils/MADP/idx_embedding_new.pth',
            "tau"               : 0.005,
            "gamma"             : 0.99,

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            'act_limit'         : 0.03,
            "device"            : parent_hp_dict['dev_rl'],
            'optim'             : 'adam',
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : parent_hp_dict['max_grad_norm'],
            "alpha"             : 0.2,
            "adaln"             : parent_hp_dict['adaln'],
            'masked'            : parent_hp_dict['masked'],
            'cmu_ri'            : parent_hp_dict['cmu_ri'],
            'gauss'             : parent_hp_dict['gauss'],
            'ca'                : parent_hp_dict['ca'],
            'learned_alpha'     : parent_hp_dict['learned_alpha'],
        }
        self.device = self.hp_dict['device']
        self.tf = Transformer(self.hp_dict)
        self.tf.to(self.device)
        self.gauss = self.hp_dict['gauss']

    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
            
        actions = self.tf.get_actions(obs, pos, deterministic)
        return actions.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_actions_batch(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).to(self.device)
            
        actions = self.tf.get_actions(obs, pos, deterministic)
        return actions.detach().cpu().numpy().mean(axis=0)

    def load_saved_policy(self, path):
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.tf.load_state_dict(expt_dict['model'])
