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
from utils.MABC.dit_core import DiffusionTransformer, EMA, _extract_into_tensor

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

class MADP:
    def __init__(self):
        self.hp_dict = {
            "exp_name"          : "MADP_1",
            "data_dir"          : "./data/rl_data",
            'warmup_iters'      : 100,
            'lr'                : 1e-4,
            'eta_min'           : 1e-6,
            'epochs'            : 100,
            'ckpt_dir'          : './madp_expt_1.pth',
            'idx_embed_loc'     : './utils/MADP/idx_embedding_128.pth',

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            "device"            : torch.device(f"cuda:0"),
            "model_dim"         : 128,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'denoising_decoder': 12},
            "dropout"           : 0,
            "max_grad_norm"     : 2.0,

            "denoising_params"  :{
                'num_train_timesteps'   : 100,
                'beta_start'        : 0.0001,
                'beta_end'          : 0.02,
                'beta_schedule'     : 'linear',
                'variance_type'     : 'fixed_small_log',
                'clip_sample'       : True ,
                'prediction_type'   : 'epsilon',
            },

            "EMA Params":{
                'update_after_step' : 0,
                'inv_gamma'         : 1.0,
                'power'             : 0.75,
                'min_value'         : 0.5,
                'max_value'         : 0.9999,
            }
        }
        self.device = self.hp_dict['device']
        self.model = DiffusionTransformer(self.hp_dict)
        self.ema_model = deepcopy(self.model).to(self.hp_dict['device'])
        self.model.to(self.hp_dict['device'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hp_dict['lr'], weight_decay=0)

        self.ema = EMA(self.ema_model, **self.hp_dict['EMA Params'])

    def actions_from_denoising_diffusion(self, x_T, states, obj_name_encs, pos, gamma=None):
        # actions get denoised from x_T --> x_t --> x_0
        actions = x_T.to(self.hp_dict['device'])
        states = states.to(self.hp_dict['device'])
        obj_name_encs = obj_name_encs.to(self.hp_dict['device'])
        pos = pos.to(self.hp_dict['device'])

        self.model.eval()
        shape = actions.shape
        with torch.no_grad():
            for i in reversed(range(self.model.denoising_params['num_train_timesteps'])):
                t = torch.tensor([i]*shape[0], device=self.model.device)
                ### p_mean_variance
                pred_noise = self.model.denoising_decoder(actions, states, obj_name_encs, pos)

                model_variance = _extract_into_tensor(self.model.posterior_variance, t, shape)
                model_log_variance = _extract_into_tensor(self.model.posterior_log_variance_clipped, t, shape)

                pred_x_start = _extract_into_tensor(self.model.sqrt_recip_alphas_cumprod, t, shape) * actions\
                            - _extract_into_tensor(self.model.sqrt_recipm1_alphas_cumprod, t, shape) * pred_noise
                
                model_mean = _extract_into_tensor(self.model.posterior_mean_coef1, t, shape) * pred_x_start\
                            + _extract_into_tensor(self.model.posterior_mean_coef2, t, shape) * actions
                
                ### p_sample
                noise = torch.randn(shape, device=self.model.device)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(shape) - 1))))
                actions = model_mean + nonzero_mask * model_variance * noise
        return actions

    def load_saved_policy(self, path):
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(expt_dict['model'])
        self.ema_model.load_state_dict(expt_dict['ema_model'])

        # madp_expt_0