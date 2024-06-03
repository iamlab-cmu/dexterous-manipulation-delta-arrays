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
import optuna
import threading

from dit_core import DiffusionTransformer, EMA, _extract_into_tensor
from madp import DataNormalizer, train
from optuna_dashboard import run_server

#TODO Tomorrow:

# Set of Expts to run:
# 1. num_samples = 5, 10, 50, 100, 250, 1000, 2000, 3000, 5000, 10000
# 2. n_epochs = 10, 50, 100, 200, 500, 1000 (inversely proportional to num_samples?)
# 3. max_grad_norm = 0.5, 1, 2, 5
# 4. num_train_timesteps = 100, 200, 500, 1000
# 5. noise_schedule = 'linear', 'squaredcos_cap_v2'
# 6. lr = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7
# 7. warmup_iters = @1%, 2%, 5%, 10% of total_iters

replay_buffer = pkl.load(open('../../data/replay_buffer.pkl', 'rb'))

rewards = replay_buffer['rew']
idxs = np.where(rewards>-5)[0]
obj_names = np.array(replay_buffer['obj_names'])

states = replay_buffer['obs'][idxs]
actions = replay_buffer['act'][idxs]
pos = replay_buffer['pos'][idxs]
obj_names = obj_names[idxs]
num_agents = replay_buffer['num_agents'][idxs]

state_scaler = DataNormalizer().fit(states)
states = state_scaler.transform(states)

action_scaler = DataNormalizer().fit(actions)
actions = action_scaler.transform(actions)

data_pkg = (states, actions, pos, num_agents, obj_names, state_scaler, action_scaler)

def objective(trial):
    num_samples = trial.suggest_int('num_samples', 5, 10000)
    n_epochs = trial.suggest_int('n_epochs', 10, 1000)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 5)
    num_train_timesteps = trial.suggest_categorical('num_train_timesteps', [100, 200, 500, 1000])
    noise_schedule = trial.suggest_categorical('noise_schedule', ['linear', 'squaredcos_cap_v2'])
    lr = trial.suggest_categorical('lr', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    eta_min = trial.suggest_categorical('eta_min', [1e-5, 1e-6, 1e-7, 1e-8])
    
    total_iters = num_samples * n_epochs
    warmup_iters = trial.suggest_categorical('warmup_iters', [0.01, 0.02, 0.05, 0.1])
    hp_dict = {
        "exp_name"          : "MADP_1",
        "data_dir"          : "../../data/rl_data",
        'warmup_iters'      : int(warmup_iters * total_iters),
        'lr'                : lr,
        'eta_min'           : eta_min,
        'epochs'            : n_epochs,
        'ckpt_dir'          : './madp_expt_1.pth',
        'idx_embed_loc'     : './idx_embedding_128.pth',

        # DiT Params:
        'state_dim'         : 6,
        'obj_name_enc_dim'  : 9,
        'action_dim'        : 2,
        "device"            : torch.device(f"cuda:3"),
        "model_dim"         : 128,
        "num_heads"         : 8,
        "dim_ff"            : 512,
        "n_layers_dict"     : {'denoising_decoder': 12},
        "dropout"           : 0,
        "max_grad_norm"     : max_grad_norm,

        "Denoising Params"  :{
            'num_train_timesteps': num_train_timesteps,
            'beta_start'         : 0.0001,
            'beta_end'           : 0.02,
            'beta_schedule'      : noise_schedule,
            'variance_type'      : 'fixed_small_log',
            'clip_sample'        : True ,
            'prediction_type'    : 'epsilon',
        },

        "EMA Params":{
            'update_after_step'  : 0,
            'inv_gamma'          : 1.0,
            'power'              : 0.75,
            'min_value'          : 0.5,
            'max_value'          : 0.9999,
        }
    }
    train(data_pkg, hp_dict)

    return result

if __name__ == '__main__':
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name="MADP Experiments")
    study.optimize(objective, n_trials=500)
    run_server(storage, host="localhost", port=8080)