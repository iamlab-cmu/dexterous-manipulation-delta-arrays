import random
import time
import wandb
import argparse

import torch

from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video

from delta_array_2dsim import DeltaArray2DSim

import utils.SAC.sac as sac
import utils.MATSAC.matsac as matsac
import utils.MATDQN.matdqn as matdqn

class DeltaArray2DEnv:
    def __init__(self, args={}, hp_dict={}):
        self.train_or_test = args.train_or_test
        self.args = args
        self.hp_dict = hp_dict
        # if not os.path.exists(f'./data/rl_data/{args.name}/ckpts'):
        #     os.makedirs(f'./data/rl_data/{args.name}/ckpts')
        
        scenario = DeltaArray2DSim()

        self.env = make_env(scenario=scenario, 
                            num_envs=args.n_envs,
                            device="cuda",  
                            seed=args.seed,
                            continuous_actions=True,
                            dict_spaces=False,
                            wrapper=None)
        self.n_envs = args.n_envs

    def train(self):
        # len(obs) = total number of agents
        obs = self.env.reset()
        print(obs[0].shape)

        action = torch.zeros((len(obs), self.n_envs, 2), device=self.env.world.device, dtype=torch.float32)
        action[0] = torch.ones((self.n_envs,2))*0.02
        for i in range(1000):
            action[0] = action[0] * -1
            print(action[0])
            for _ in range(100):
                obs, rews, dones, info = self.env.step(action)
                frame = self.env.render(mode="human")
            # self.env.render(
            #         mode="rgb_array",
            #         agent_index_focus=None,
            #         visualize_when_rgb=True,
            #     )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Args for sim/real test/train")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--train_or_test", type=str, default="test")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=float, default=69)
    
    args = parser.parse_args()
    # hp_dict = {
    #     # General RL Hyperparameters:
    #     "exp_name"    :args.name,
    #     "data_dir"    :"./data/rl_data",
    #     "tau"         :0.005,
    #     "gamma"       :0.99,
    #     "q_lr"        :1e-3,
    #     "pi_lr"       :1e-3,
    #     "alpha"       :0.2,
    #     "replay_size" :500000,
    #     'seed'        :69420,
    #     'optim'       :'sgd',
    #     "batch_size"  :args.bs,
    #     "exploration_cutoff": args.expl,
    #     "dev_sim"       : torch.device(f"cuda:{args.dev_sim}"),
    #     "dev_rl"        : torch.device(f"cuda:{args.dev_rl}"),
    #     "dont_log"      : args.dont_log,

    #     # Single Agent Part:
    #     'act_dim'       :2,
    #     'state_dim'     :6,

    #     # Multi Agent Part:
    #     'state_dim'     : 6,
    #     "model_dim"     : 128,
    #     "num_heads"     : 8,
    #     "dim_ff"        : 64,
    #     "n_layers_dict" :{'encoder': 3, 'actor': 3, 'critic': 3},
    #     "dropout"       : 0,
    #     "add_vs_data"   : args.add_vs_data,
    #     "ratio"         : args.vs_data,
    # }

    delta_array_2d_env = DeltaArray2DEnv(args)
    delta_array_2d_env.train()