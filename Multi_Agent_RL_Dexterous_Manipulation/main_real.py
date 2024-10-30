import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse
import wandb

import torch

import delta_array_real_new as delta_array_real
import utils.SAC.sac as sac
# import utils.DDPG.ddpg as ddpg
# import utils.MASAC.masac as masac
import utils.MATSAC.matsac as matsac
import utils.MATDQN.matdqn as matdqn
import utils.MADP.madptest as madp0
import utils.MADP.mabc_test as mabc
from utils.MADP.madp import DataNormalizer
import config.assets.obj_dict_real as obj_dict

from utils.openai_utils.run_utils import setup_logger_kwargs


class DeltaArrayRealEnvironment():
    def __init__(self, train_or_test="test"):
        self.train_or_test = train_or_test
        self.args = args
        single_agent_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 4},}
        ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'pi_obs_space'  : {'dim': 6},
                    'q_obs_space'   : {'dim': 6},
                    "max_agents"    : 64,}
        
        self.hp_dict = {
                "exp_name"          : args.name,
                "data_dir"          : "./data/rl_data",
                "tau"               : 0.005,
                "gamma"             : 0.99,
                "q_lr"              : 1e-2,
                "pi_lr"             : 1e-2,
                "q_eta_min"         : self.args.q_etamin,
                "pi_eta_min"        : self.args.pi_etamin,
                "eta_min"           : self.args.q_etamin,
                "alpha"             : 0.2,
                "replay_size"       : 500000,
                'seed'              : 69420,
                'optim'             : 'sgd',
                'epsilon'           : 0.9,
                "batch_size"        : self.args.bs,
                "exploration_cutoff": self.args.expl,
                "robot_frame"       : self.args.robot_frame,
                "infer_every"       : 4000,
                "inference_length"  : 10,
                'save_videos'       : args.save_vid,

                # Multi Agent Part Below:
                'state_dim'         : 6,
                "dev_sim"           : torch.device(f"cuda:{self.args.dev_sim}"),
                "dev_rl"            : torch.device(f"cuda:{self.args.dev_rl}"),
                "model_dim"         : 256,
                "num_heads"         : 8,
                "dim_ff"            : 128,
                "n_layers_dict"     : {'encoder': 10, 'actor': 10, 'critic': 10},
                "dropout"           : 0,
                "max_grad_norm"     : 1.0,
                "adaln"             : self.args.adaln,
                "delta_array_size"  : [8,8],
                "dont_log"          : True,
                'print_summary'     : self.args.print_summary,
                'vis_servo'         : self.args.vis_servo,
                'test_traj'         : self.args.test_traj,
                'obj_name'          : self.args.obj,
                'traditional'       : self.args.traditional,
            }

        logger_kwargs = {}
        if self.train_or_test=="train":
            if not self.hp_dict["dont_log"]:
                logger_kwargs = setup_logger_kwargs(self.hp_dict['exp_name'], 69420, data_dir=self.hp_dict['data_dir'])

        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, logger_kwargs, ma=False, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')

        if self.args.algo=="MATSAC":
            self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
        elif self.args.algo=="MATDQN":
            self.pushing_agent = matdqn.MATDQN(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
        elif self.args.algo=="MADP":
            self.pushing_agent = madp0.MADP()
        elif self.args.algo=="MABC":
            self.pushing_agent = mabc.MABC()

        if (self.train_or_test=="test") and (not self.args.behavior_cloning):
            self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/pyt_save/model.pt')
        elif self.args.behavior_cloning:
            self.pushing_agent.load_saved_policy(f'./utils/MADP/{args.name}.pth')

        self.objects = obj_dict.get_obj_dict()
        self.delta_array = delta_array_real.DeltaArrayReal([self.grasping_agent, self.pushing_agent], self.objects, self.hp_dict)

    def run(self):
        if self.args.test_traj:
            self.delta_array.trajectory_rollout()
        elif self.args.vis_servo:
            self.delta_array.visual_servoing()
        elif self.args.behavior_cloning:
            if self.args.algo=="MADP":
                self.delta_array.diffusion_policy()
            elif self.args.algo=="MABC":
                self.delta_array.bc_policy()
        else:
            self.delta_array.test_grasping_policy(reset_after=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for sim/real test/train")

    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    parser.add_argument("-bc", "--behavior_cloning", action="store_true", help="True for Testing BC Policies")
    parser.add_argument("-n", "--name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-obj_name", "--obj_name", type=str, default="disc", help="Object Name in env.yaml")
    parser.add_argument("-dev_sim", "--dev_sim", type=int, default=0, help="Device for Sim")
    parser.add_argument("-dev_rl", "--dev_rl", type=int, default=0, help="Device for RL")
    parser.add_argument("-bs", "--bs", type=int, default=256, help="Batch Size")
    parser.add_argument("-expl", "--expl", type=int, default=512, help="Exploration Cutoff")
    parser.add_argument("-algo", "--algo", type=str, default="MATSAC", help="RL Algorithm")
    parser.add_argument("-rf", "--robot_frame", action="store_true", help="Robot Frame Yes or No")
    parser.add_argument("-print", "--print_summary", action="store_true", help="Print Summary and Store in Pickle File")
    parser.add_argument("-pilr", "--pilr", type=float, default=1e-2, help="% of data to use for visual servoing")
    parser.add_argument("-qlr", "--qlr", type=float, default=1e-2, help="% of data to use for visual servoing")
    parser.add_argument("-adaln", "--adaln", action="store_true", help="Use AdaLN Zero Transformer")
    parser.add_argument("-q_etamin", "--q_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-pi_etamin", "--pi_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-savevid", "--save_vid", action="store_true", help="Save Videos at inference")
    parser.add_argument("-fingers4", "--fingers4", action="store_true", help="Use simplified setup with only 4 fingers")
    parser.add_argument("-XX", "--donothing", action="store_true", help="Do nothing to test sim")
    parser.add_argument("-gradnorm", "--gradnorm", type=float, default=1.0, help="Grad norm for training")
    parser.add_argument("-test_traj", "--test_traj", action="store_true", help="Test on trajectories")
    parser.add_argument("-obj", "--obj", type=str, default="trapezium", help="Object Name")
    parser.add_argument("-trad", "--traditional", action="store_true", help="Traditional Vision Pipeline")
    args = parser.parse_args()

    if args.vis_servo and not args.test:
        parser.error("--vis_servo requires --test")
        sys.exit(1)
    if args.name=="HAKUNA":
        parser.error("Expt name is required for training")
        sys.exit(1)

    train_or_test = "test" if args.test else "train"
    
    delta_array_real.start_capture_thread()
    env = DeltaArrayRealEnvironment(train_or_test)
    env.run()