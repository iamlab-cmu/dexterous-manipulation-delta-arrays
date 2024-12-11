import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse
from autolab_core import YamlConfig, RigidTransform
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset, GymURDFAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, rpy_to_quat
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera
import wandb
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

import src.delta_array_sim as delta_array_sim
import src.delta_array_simplified as delta_array_simplified
import src.delta_array_sim_image as delta_array_sim_image
import src.delta_array_real as delta_array_real
import src.delta_array_rope as delta_array_rope
import utils.SAC.sac as sac
# import utils.DDPG.ddpg as ddpg
# import utils.MASAC.masac as masac
# import utils.MATSAC.matsac as matsac
import utils.MAT.mat as matsacOGOG
import utils.MATSAC.matsac_no_autoreg as matsac
import utils.MATDQN.matdqn as matdqn
import utils.MADP.madptest as madp0
import utils.MADP.mabc_test as mabc
import utils.MADP.mabc_finetune as mabc_finetune
from utils.MADP.madp import DataNormalizer
import config.assets.obj_dict as obj_dict

from utils.openai_utils.run_utils import setup_logger_kwargs

class DeltaArraySimEnvironment():
    def __init__(self, yaml_path, run_no, train_or_test="test", args={}):
        self.train_or_test = train_or_test
        self.args = args
        gym = gymapi.acquire_gym()
        self.run_no = run_no
        self.cfg = YamlConfig(yaml_path)
        self.cfg['scene']['n_envs'] = self.args.num_expts
        self.cfg['scene']['gui'] = self.args.gui
        self.cfg['scene']['device']['compute'] = self.args.dev_sim
        self.cfg['scene']['device']['graphics'] = self.args.dev_sim
        self.scene = GymScene(self.cfg['scene'])
        if not os.path.exists('./data/manip_data'):   
            os.makedirs('./data/manip_data')

        # self.obj_name = args.obj_name
        # self.object = GymBoxAsset(self.scene, **self.cfg[self.obj_name]['dims'], 
        #                     shape_props=self.cfg[self.obj_name]['shape_props'], 
        #                     rb_props=self.cfg[self.obj_name]['rb_props'],
        #                     asset_options=self.cfg[self.obj_name]['asset_options'])
        
        # self.object = GymURDFAsset(self.cfg[self.obj_name]['urdf_path'], self.scene, 
        #                     shape_props=self.cfg[self.obj_name]['shape_props'], 
        #                     rb_props=self.cfg[self.obj_name]['rb_props'],
        #                     asset_options=self.cfg[self.obj_name]['asset_options'],
        #                     assets_root=Path('config'))

        self.objects = obj_dict.get_obj_dict(GymURDFAsset, self.scene, self.cfg)

        self.table = GymURDFAsset(self.cfg['table']['urdf_path'], self.scene,
                        asset_options=self.cfg['table']['asset_options'],
                        rb_props=self.cfg['table']['rb_props'],
                        shape_props=self.cfg['table']['shape_props'],
                        assets_root=Path('config'))

        # # Left Top fiducial marker
        # self.fiducial_lt = GymBoxAsset(self.scene, **self.cfg['fiducial']['dims'], 
        #                     shape_props=self.cfg['fiducial']['shape_props'], 
        #                     rb_props=self.cfg['fiducial']['rb_props'],
        #                     asset_options=self.cfg['fiducial']['asset_options'])
        # # Right Bottom fiducial marker
        # self.fiducial_rb = GymBoxAsset(self.scene, **self.cfg['fiducial']['dims'], 
        #                     shape_props=self.cfg['fiducial']['shape_props'], 
        #                     rb_props=self.cfg['fiducial']['rb_props'],
        #                     asset_options=self.cfg['fiducial']['asset_options'])
        
        if not os.path.exists(f'./data/rl_data/{args.name}/pyt_save'):
            os.makedirs(f'./data/rl_data/{args.name}/pyt_save')
            os.makedirs(f'./data/rl_data/{args.name}/videos')

        if not os.path.exists(f'./data/videos/{args.name}'):
            os.makedirs(f'./data/videos/{args.name}')

        single_agent_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 4},}
        ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'pi_obs_space'  : {'dim': 6},
                    'q_obs_space'   : {'dim': 6},
                    "max_agents"    : 64,}
        
        simplified_ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 8},
                    'observation_space'  : {'dim': 24},
                    "max_agents"    : 4}
        
        self.hp_dict = {
                # Env Params
                "exp_name"          : args.name,
                'algo'              : self.args.algo,
                'data_type'         : self.args.data_type,
                'vis_servo'         : self.args.vis_servo,
                'test_traj'         : self.args.test_traj,
                "dont_log"          : self.args.dont_log,
                "replay_size"       : 500001,
                'seed'              : 69420,
                "data_dir"          : "./data/rl_data",
                
                # RL params
                "tau"               : 0.005,
                "gamma"             : 0.99,
                "q_lr"              : self.args.qlr,
                "pi_lr"             : self.args.pilr,
                'obj_name_enc_dim'  : 9,
                "q_eta_min"         : self.args.q_etamin,
                "pi_eta_min"        : self.args.pi_etamin,
                "eta_min"           : self.args.q_etamin,
                "alpha"             : 0.2,
                'optim'             : self.args.optim,
                'epsilon'           : 1.0,
                "batch_size"        : self.args.bs,
                "warmup_epochs"     : self.args.warmup,
                "robot_frame"       : self.args.robot_frame,
                "infer_every"       : 4000,
                "inference_length"  : 10,
                'save_videos'       : args.save_vid,
                'act_limit'         : 0.03,

                # Multi Agent Part Below:
                'state_dim'         : 6,
                'action_dim'        : 2,
                "dev_sim"           : torch.device(f"cuda:{self.args.dev_sim}"),
                "dev_rl"            : torch.device(f"cuda:{self.args.dev_rl}"),
                "model_dim"         : 256,
                "num_heads"         : 8,
                "dim_ff"            : 128,
                "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
                "dropout"           : 0,
                "max_grad_norm"     : self.args.gradnorm,
                "adaln"             : self.args.adaln,
                "delta_array_size"  : [8,8],
                "add_vs_data"       : self.args.add_vs_data,
                "vs_ratio"          : self.args.vs_data,
                'print_summary'     : self.args.print_summary,
                'masked'            : not self.args.unmasked,
                'cmu_ri'            : self.args.cmuri,
                'gauss'             : self.args.gauss,
                'ca'                : self.args.ca,
                'learned_alpha'     : self.args.la,
                
                # Test Traj Params
                'test_algos'        : ['MABC', 'Vis Servo', 'Random', 'MATSAC'], #, 'MABC Finetuned'
            }
        
        logger_kwargs = {}
        if self.train_or_test=="train":
            if not self.hp_dict["dont_log"]:
                logger_kwargs = setup_logger_kwargs(self.hp_dict['exp_name'], 69420, data_dir=self.hp_dict['data_dir'])
                # writer = SummaryWriter(log_dir=f"./tensorboard/{self.hp_dict['exp_name']}")
                wandb.init(project="MARL_Dexterous_Manipulation",
                        config=self.hp_dict,
                        name = self.hp_dict['exp_name'])

        # self.agent = ddpg.DDPG(env_dict, self.hp_dict, logger_kwargs)
        # self.agent = reinforce.REINFORCE(env_dict, 3e-3)
        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, logger_kwargs, ma=False, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')


        if self.args.test_traj:
            self.pushing_agent = {
                "Random" : None,
                "Vis Servo" : None,
                "MATSAC" : matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="test"),
                "MABC" : mabc.MABC(self.hp_dict),
                # "MABC Finetuned" : mabc_finetune.MABC_Finetune(self.hp_dict),
            }
            self.pushing_agent["MATSAC"].load_saved_policy('./data/rl_data/matsac_FINAL/pyt_save/model.pt')
            self.pushing_agent["MABC"].load_saved_policy('./utils/MADP/mabc_final.pth')
            # self.pushing_agent["MABC Finetuned"].load_saved_policy('./utils/MADP/MABC_Finetuned.pth')
        else:
            if self.args.algo=="MATSAC":
                self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
            elif self.args.algo=="MATSAC_OGOG":
                self.pushing_agent = matsacOGOG.MATSAC_OGOG(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
            elif self.args.algo=="MATDQN":
                self.pushing_agent = matdqn.MATDQN(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
            elif self.args.algo=="SAC":
                self.pushing_agent = sac.SAC(simplified_ma_env_dict, self.hp_dict, logger_kwargs, ma=True, train_or_test="train")
            elif self.args.algo=="MADP":
                self.pushing_agent = madp0.MADP()
            elif self.args.algo=="MABC":
                self.pushing_agent = mabc.MABC()
            elif self.args.algo=="MABC_Finetune":
                self.pushing_agent = mabc_finetune.MABC_Finetune(self.hp_dict)

            if (self.train_or_test=="test") and (not self.args.behavior_cloning):
                # self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/{args.name}_s69420/pyt_save/model.pt')
                self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/pyt_save/model.pt')
            elif self.args.behavior_cloning:
                self.pushing_agent.load_saved_policy(f'./utils/MADP/{args.name}.pth')
        
        # if self.args.data_type=="finger4":
        #     self.fingers = delta_array_simplified.DeltaArraySim(self.scene, self.cfg, self.objects, self.table, None, None, [self.grasping_agent, self.pushing_agent], self.hp_dict, num_tips = [8,8], max_agents=ma_env_dict['max_agents'])
        # elif self.args.data_type=="images":
        #     self.fingers = delta_array_sim_image.DeltaArraySim(self.scene, self.cfg, self.objects, self.table, None, None, [self.grasping_agent, self.pushing_agent], self.hp_dict, num_tips = [8,8], max_agents=ma_env_dict['max_agents'])
        # elif self.args.rope:
        #     self.fingers = delta_array_rope.DeltaArraySim(self.scene, self.cfg, self.objects, self.table, None, None, [self.grasping_agent, self.pushing_agent], self.hp_dict, num_tips = [8,8], max_agents=ma_env_dict['max_agents'])
        # else:
        self.fingers = delta_array_sim.DeltaArraySim(self.scene, self.cfg, self.objects, self.table, None, None, [self.grasping_agent, self.pushing_agent], self.hp_dict, num_tips = [8,8], max_agents=ma_env_dict['max_agents'])
        
        
        self.cam = GymCamera(self.scene, cam_props = self.cfg['camera'])
        rot = RigidTransform.x_axis_rotation(np.deg2rad(0))@RigidTransform.z_axis_rotation(np.deg2rad(-90))
        self.cam_offset_transform = RigidTransform_to_transform(RigidTransform(
            rotation=rot,
            translation = np.array([0.13125, 0.1407285, 0.65])
            # translation = np.array([2, 2, 0.65])
        ))
        self.cam_name = 'hand_cam0'
        self.fingers.cam = self.cam 
        self.fingers.cam_name = self.cam_name

        self.scene.setup_all_envs(self.setup_scene)
        self.setup_objects()

    def setup_scene(self, scene, _):
        self.fingers.add_asset(scene)
        scene.add_asset("table", self.table, gymapi.Transform())
        if self.args.rope:
            self.rope = GymURDFAsset('assets/rope.urdf', scene, 
                    shape_props=self.cfg['rope']['shape_props'], 
                    rb_props=self.cfg['rope']['rb_props'],
                    asset_options=self.cfg['rope']['asset_options'],
                    assets_root=Path('config'))
            scene.add_asset('rope', self.rope, gymapi.Transform(p=gymapi.Vec3(0.13125, 0.1407285, 1.5), r=gymapi.Quat(0.8509035, 0, 0, 0.525322)))
        else:    
            for obj_name in self.objects.keys():
                obj, object_p, _, object_r = self.objects[obj_name]
                scene.add_asset(obj_name, obj, gymapi.Transform(p=object_p, r=object_r))
        # scene.add_asset("fiducial_lt", self.fiducial_lt, gymapi.Transform()) 
        # scene.add_asset("fiducial_rb", self.fiducial_rb, gymapi.Transform()) 
        scene.add_standalone_camera(self.cam_name, self.cam, self.cam_offset_transform)
        scene.gym.set_light_parameters(scene.sim, 0, gymapi.Vec3(1, 1, 1),gymapi.Vec3(1, 1, 1),gymapi.Vec3(0, -1, -1))

    def setup_objects(self):
        for i in self.scene.env_idxs:
            self.fingers.set_attractor_handles(i)

        table_transforms = [gymapi.Transform(p=gymapi.Vec3(0,0,0.5)) for _ in range(self.scene.n_envs)]
        # fiducial_rb = [gymapi.Transform(p=gymapi.Vec3(-0.2035, -0.06, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_lt = [gymapi.Transform(p=gymapi.Vec3(0.303107 + 0.182, 0.2625 + 0.06, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_rb = [gymapi.Transform(p=gymapi.Vec3(0 - 0.0612, 0 - 0.2085, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_lt = [gymapi.Transform(p=gymapi.Vec3(0.2625 + 0.062, 0.303107 + 0.1855, 1.0052)) for _ in range(self.scene.n_envs)]
        for env_idx in self.scene.env_idxs:
            self.table.set_rb_transforms(env_idx, 'table', [table_transforms[env_idx]])
            self.fingers.obj_name[env_idx] = "rope" if self.args.rope else "disc" 
            self.fingers.object[env_idx] = self.rope if self.args.rope else self.objects[self.fingers.obj_name[env_idx]][0]
            
            if self.hp_dict['test_traj']:
                exit_bool, pos = self.fingers.set_traj_pose(env_idx, goal=True)
                # self.fingers.tracked_trajs[self.fingers.obj_name[env_idx]]['traj'].append(pos)
                # self.fingers.tracked_trajs[self.fingers.obj_name[env_idx]]['error'].append((np.linalg.norm(self.fingers.goal_pose[env_idx][:2] - pos[:2]), self.fingers.angle_difference(pos[2], self.fingers.goal_pose[env_idx, 2])))
            else:
                self.fingers.set_block_pose(env_idx, goal=True)
            self.fingers.set_all_fingers_pose(env_idx)
            
            # self.fiducial_lt.set_rb_transforms(env_idx, "fiducial_lt", [fiducial_lt[env_idx]])
            # self.fiducial_rb.set_rb_transforms(env_idx, "fiducial_rb", [fiducial_rb[env_idx]])

    def run(self):
        # self.scene.run(policy=self.fingers.sim_test)
        if self.train_or_test=="train":
            self.scene.run(policy=self.fingers.inverse_dynamics)
        else:
            if self.args.donothing:
                self.scene.run(policy=self.fingers.do_nothing)
            elif self.args.vis_servo:
                self.scene.run(policy=self.fingers.visual_servoing)
                # self.scene.run(policy=self.fingers.do_nothing)
            elif self.args.behavior_cloning:
                self.scene.run(policy=self.fingers.test_diffusion_policy)
            else:
                # self.scene.run(policy=self.fingers.test_learned_policy)
                if self.args.test_traj:
                    self.scene.run(policy=self.fingers.test_trajs_algos)
                else:
                    self.scene.run(policy=self.fingers.compare_policies)

class DeltaArrayRealEnvironment():
    def __init__(self, train_or_test="test"):
        self.train_or_test = train_or_test
        env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 4}}
        self.hp_dict = {
                "tau"         :0.005,
                "gamma"       :0.99,
                "q_lr"        :1e-4,
                "pi_lr"       :1e-4,
                "alpha"       :0.2,
                "replay_size" :200000,
                'seed'        :3
            }

        if self.train_or_test=="train":
            logger_kwargs = setup_logger_kwargs("ddpg_expt_0", 69420, data_dir="./data/rl_data")
        else:
            logger_kwargs = {}
        # self.agent = ddpg.DDPG(env_dict, self.hp_dict, logger_kwargs)
        self.agent = sac.SAC(env_dict, self.hp_dict, logger_kwargs=logger_kwargs, train_or_test=self.train_or_test)
        # self.agent = reinforce.REINFORCE(env_dict, 3e-3)
        if self.train_or_test=="test":
            self.agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')
        self.delta_array = delta_array_real.DeltaArrayReal(None, None, agent=self.agent)

    def run(self):
        self.delta_array.test_grasping_policy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for sim/real test/train")

    parser.add_argument("-r", "--real", action="store_true", help="True for Real Robot Expt")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    parser.add_argument("-bc", "--behavior_cloning", action="store_true", help="True for Testing Diff Policy")
    parser.add_argument("-nexp", "--num_expts", type=int, default=1, help="Number of Experiments to run")
    parser.add_argument("-gui", "--gui", action="store_true", help="True for GUI")
    parser.add_argument("-avsd", "--add_vs_data", action="store_true", help="True for adding visual servoing data")
    parser.add_argument("-vsd", "--vs_data", type=float, help="[0 to 1] ratio of data to use for visual servoing")
    parser.add_argument("-n", "--name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-obj_name", "--obj_name", type=str, default="disc", help="Object Name in env.yaml")
    parser.add_argument("-dontlog", "--dont_log", action="store_true", help="Don't Log Experiment")
    parser.add_argument("-dev_sim", "--dev_sim", type=int, default=5, help="Device for Sim")
    parser.add_argument("-dev_rl", "--dev_rl", type=int, default=1, help="Device for RL")
    parser.add_argument("-bs", "--bs", type=int, default=256, help="Batch Size")
    parser.add_argument("-warmup", "--warmup", type=int, default=5000, help="Exploration Cutoff")
    parser.add_argument("-algo", "--algo", type=str, default="MATSAC", help="RL Algorithm")
    parser.add_argument("-rf", "--robot_frame", action="store_true", help="Robot Frame Yes or No")
    parser.add_argument("-print", "--print_summary", action="store_true", help="Print Summary and Store in Pickle File")
    parser.add_argument("-pilr", "--pilr", type=float, default=1e-2, help="% of data to use for visual servoing")
    parser.add_argument("-qlr", "--qlr", type=float, default=1e-2, help="% of data to use for visual servoing")
    parser.add_argument("-adaln", "--adaln", action="store_true", help="Use AdaLN Zero Transformer")
    parser.add_argument("-q_etamin", "--q_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-pi_etamin", "--pi_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-savevid", "--save_vid", action="store_true", help="Save Videos at inference")
    parser.add_argument("-data", "--data_type", type=str, default=None, help="Use simplified setup with only 4 fingers")
    parser.add_argument("-XX", "--donothing", action="store_true", help="Do nothing to test sim")
    parser.add_argument("-gradnorm", "--gradnorm", type=float, default=1.0, help="Grad norm for training")
    parser.add_argument("-test_traj", "--test_traj", action="store_true", help="Test on trajectories")
    parser.add_argument("-cmuri", "--cmuri", action="store_true", help="Set to use CMU RI trajectory")
    parser.add_argument("-unmasked", "--unmasked", action="store_true", help="Unmasked Attention Layers")
    parser.add_argument("-gauss", "--gauss", action="store_true", help="Use Gaussian Final Layers")
    parser.add_argument("-rope", "--rope", action="store_true", help="To rope or not to rope")
    parser.add_argument("-optim", "--optim", type=str, default="adam", help="Optimizer to use adam vs sgd")
    parser.add_argument("-ca", "--ca", action="store_true", help="compensate for Actions in reward function")
    parser.add_argument("-la", "--la", action="store_true", help="Is Alpha Learned?")
    args = parser.parse_args()

    if args.vis_servo and not args.test:
        parser.error("--vis_servo requires --test")
        sys.exit(1)
    if args.name=="HAKUNA":
        parser.error("Expt name is required for training")
        sys.exit(1)

    train_or_test = "test" if args.test else "train"
    if not args.real:
        yaml_path = './config/env.yaml'
        run_no = 0
        env = DeltaArraySimEnvironment(yaml_path, run_no, train_or_test, args)
        env.run()
    else:
        env = DeltaArrayRealEnvironment(train_or_test)
        env.run()