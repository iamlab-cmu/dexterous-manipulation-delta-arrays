import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse
import wandb
from autolab_core import YamlConfig, RigidTransform
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset, GymURDFAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, rpy_to_quat
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

import delta_array_sim
import delta_array_real
import utils.SAC.sac as sac
import utils.DDPG.ddpg as ddpg
import utils.MASAC.masac as masac
import utils.MATSAC.matsac as matsac

from utils.openai_utils.run_utils import setup_logger_kwargs
device = torch.device("cuda:1")

class DeltaArraySimEnvironment():
    def __init__(self, yaml_path, run_no, train_or_test="test", args={}):
        self.train_or_test = train_or_test
        self.args = args
        gym = gymapi.acquire_gym()
        self.run_no = run_no
        self.cfg = YamlConfig(yaml_path)
        self.cfg['scene']['n_envs'] = self.args.num_expts
        self.cfg['scene']['gui'] = self.args.gui
        self.scene = GymScene(self.cfg['scene'])
        if not os.path.exists('./data/manip_data'):   
            os.makedirs('./data/manip_data')

        self.obj_name = args.obj_name
        # self.object = GymBoxAsset(self.scene, **self.cfg[self.obj_name]['dims'], 
        #                     shape_props=self.cfg[self.obj_name]['shape_props'], 
        #                     rb_props=self.cfg[self.obj_name]['rb_props'],
        #                     asset_options=self.cfg[self.obj_name]['asset_options'])
        
        self.object = GymURDFAsset(self.cfg[self.obj_name]['urdf_path'], self.scene, 
                            shape_props=self.cfg[self.obj_name]['shape_props'], 
                            rb_props=self.cfg[self.obj_name]['rb_props'],
                            asset_options=self.cfg[self.obj_name]['asset_options'],
                            assets_root=Path('config'))

        self.table = GymURDFAsset(self.cfg['table']['urdf_path'], self.scene,
                        asset_options=self.cfg['table']['asset_options'],
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
        
        # self.model = resnet18(weights="ResNet18_Weights.DEFAULT")
        # self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        # self.model.eval()
        # self.model = self.model.to(device)
        # self.transform = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        if not os.path.exists(f'./data/rl_data/{args.name}/pyt_save'):
            os.makedirs(f'./data/rl_data/{args.name}/pyt_save')

        single_agent_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 4},}
        ma_env_dict = {'action_space': {'low': -0.06, 'high': 0.06, 'dim': 2},
                    'pi_obs_space': {'dim': 6},
                    'q_obs_space': {'dim': 6},
                    "max_agents"    :64,}
        self.hp_dict = {
                "exp_name": args.name,
                "data_dir": "./data/rl_data",
                "tau"         :0.005,
                "gamma"       :0.99,
                "q_lr"        :1e-4,
                "pi_lr"       :1e-4,
                "alpha"       :0.2,
                "replay_size" :500000,
                'seed'        :69420,
                "batch_size"  :256,
                "exploration_cutoff": 512,

                # Multi Agent Part Below:
                'state_dim': 6,
                "device": torch.device("cuda:0"),
                "model_dim": 128,
                "num_heads": 8,
                "dim_ff": 64,
                "n_layers_dict":{'encoder': 3, 'actor': 3, 'critic': 2},
                "dropout": 0,
                "delta_array_size": [8,8],
                "add_vs_data": self.args.add_vs_data,
                "ratio": self.args.vs_data,
                "dont_log": self.args.dont_log,
            }
        
        logger_kwargs = {}
        if self.train_or_test=="train":
            if not self.hp_dict["dont_log"]:
                logger_kwargs = setup_logger_kwargs(self.hp_dict['exp_name'], 69420, data_dir=self.hp_dict['data_dir'])
                wandb.init(project="MARL_Dexterous_Manipulation",
                        config=self.hp_dict,
                        name = self.hp_dict['exp_name'])

        # self.agent = ddpg.DDPG(env_dict, self.hp_dict, logger_kwargs)
        # self.agent = reinforce.REINFORCE(env_dict, 3e-3)
        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, logger_kwargs, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')

        # self.pushing_agent = masac.MASAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
        self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")

        if self.train_or_test=="test":
            # self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/{args.name}_s69420/pyt_save/model.pt')
            self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/pyt_save/model.pt')
        
        self.fingers = delta_array_sim.DeltaArraySim(self.scene, self.cfg, self.object, self.obj_name, None, None, [self.grasping_agent, self.pushing_agent], self.hp_dict, num_tips = [8,8], max_agents=ma_env_dict['max_agents'])
        
        
        self.cam = GymCamera(self.scene, cam_props = self.cfg['camera'])
        rot = RigidTransform.x_axis_rotation(np.deg2rad(0))@RigidTransform.z_axis_rotation(np.deg2rad(-90))
        self.cam_offset_transform = RigidTransform_to_transform(RigidTransform(
            rotation=rot,
            translation = np.array([0.13125, 0.1407285, 0.65])
        ))
        self.cam_name = 'hand_cam0'
        self.fingers.cam = self.cam 
        self.fingers.cam_name = self.cam_name

        self.scene.setup_all_envs(self.setup_scene)
        self.setup_objects()

    def setup_scene(self, scene, _):
        # we'll sample block poses later
        self.fingers.add_asset(scene)
        # Add either rigid body or soft body as an asset to the scene
        scene.add_asset(self.obj_name, self.object, gymapi.Transform()) 
        scene.add_asset("table", self.table, gymapi.Transform())
        # scene.add_asset("fiducial_lt", self.fiducial_lt, gymapi.Transform()) 
        # scene.add_asset("fiducial_rb", self.fiducial_rb, gymapi.Transform()) 
        scene.add_standalone_camera(self.cam_name, self.cam, self.cam_offset_transform)
        scene.gym.set_light_parameters(scene.sim, 0, gymapi.Vec3(1, 1, 1),gymapi.Vec3(1, 1, 1),gymapi.Vec3(0, -1, -1))

    def setup_objects(self):
        for i in self.scene.env_idxs:
            self.fingers.set_attractor_handles(i)

        object_p = gymapi.Vec3(0.13125, 0.1407285, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
        # object_r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)
        object_r = gymapi.Quat(0, 0, 0, 1)
        object_transforms = [gymapi.Transform(p=object_p, r=object_r) for _ in range(self.scene.n_envs)]
        table_transforms = [gymapi.Transform(p=gymapi.Vec3(0,0,0.5)) for _ in range(self.scene.n_envs)]
        # fiducial_rb = [gymapi.Transform(p=gymapi.Vec3(-0.2035, -0.06, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_lt = [gymapi.Transform(p=gymapi.Vec3(0.303107 + 0.182, 0.2625 + 0.06, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_rb = [gymapi.Transform(p=gymapi.Vec3(0 - 0.0612, 0 - 0.2085, 1.0052)) for _ in range(self.scene.n_envs)]
        # fiducial_lt = [gymapi.Transform(p=gymapi.Vec3(0.2625 + 0.062, 0.303107 + 0.1855, 1.0052)) for _ in range(self.scene.n_envs)]
        for env_idx in self.scene.env_idxs:
            self.table.set_rb_transforms(env_idx, 'table', [table_transforms[env_idx]])
            
            yaw = np.arctan2(2*(object_r.w*object_r.z + object_r.x*object_r.y), 1 - 2*(object_r.x**2 + object_r.y**2))
            T = np.array((np.random.uniform(0.009, 0.21), np.random.uniform(0.005, 0.25)))
            self.fingers.goal_pose[env_idx] = np.array([*T, yaw])
            
            self.object.set_rb_transforms(env_idx, self.obj_name, [object_transforms[env_idx]])
            self.fingers.set_all_fingers_pose(env_idx)
            # self.fingers.set_block_pose(env_idx)
            # self.fiducial_lt.set_rb_transforms(env_idx, "fiducial_lt", [fiducial_lt[env_idx]])
            # self.fiducial_rb.set_rb_transforms(env_idx, "fiducial_rb", [fiducial_rb[env_idx]])

    def run(self):
        # self.scene.run(policy=self.fingers.sim_test)
        if self.train_or_test=="train":
            self.scene.run(policy=self.fingers.inverse_dynamics)
        else:
            if self.args.vis_servo:
                self.scene.run(policy=self.fingers.visual_servoing)
                # self.scene.run(policy=self.fingers.do_nothing)
            else:
                self.scene.run(policy=self.fingers.test_learned_policy)

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
                "replay_size" :100000,
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
    parser.add_argument("-nexp", "--num_expts", type=int, default=1, help="Number of Experiments to run")
    parser.add_argument("-gui", "--gui", action="store_true", help="True for GUI")
    parser.add_argument("-avsd", "--add_vs_data", action="store_true", help="True for adding visual servoing data")
    parser.add_argument("-vsd", "--vs_data", type=float, help="% of data to use for visual servoing")
    parser.add_argument("-n", "--name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-on", "--obj_name", type=str, default="disc", help="Object Name in env.yaml")
    parser.add_argument("-dontlog", "--dont_log", action="store_true", help="Don't Log to Wandb")
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