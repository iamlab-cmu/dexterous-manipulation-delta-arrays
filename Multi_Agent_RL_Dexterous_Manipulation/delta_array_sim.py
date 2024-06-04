import numpy as np
import signal
import sys
import time
import pickle as pkl
import time
import random
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import networkx as nx
from PIL import Image
from scipy.spatial.transform import Rotation as R
import wandb
np.set_printoptions(precision=4)

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

from autolab_core import YamlConfig, RigidTransform, PointCloud
from visualization.visualizer3d import Visualizer3D as vis3d

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset
from isaacgym_utils.math_utils import RigidTransform_to_transform, quat_to_rpy, rpy_to_quat
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import utils.nn_helper as helper
# import utils.log_utils as log_utils
import utils.rl_utils as rl_utils
# import utils.SAC.sac as sac
# import utils.SAC.reinforce as reinforce
import utils.geometric_utils as geom_utils

class DeltaArraySim:
    def __init__(self, scene, cfg, objs, table, img_embed_model, transform, agents, hp_dict, num_tips = [8,8], max_agents=64):
        """ Main Vars """
        self.scene = scene
        self.cfg = cfg
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        self.obj_dict = objs
        self.table = table
        self.obj_names = list(self.obj_dict.keys())
        self.obj_name = {}
        self.object = {}
        self.max_agents = max_agents
        self.hp_dict = hp_dict
        
        # Introduce delta robot EE in the env
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.fingertips[i][j] = (GymCapsuleAsset(scene, **cfg['capsule']['dims'],
                    shape_props=cfg['capsule']['shape_props'],
                    rb_props=cfg['capsule']['rb_props'],
                    asset_options=cfg['capsule']['asset_options']
                ))
        """ Data Collection Vars """
        self.img_size = np.array((1080, 1920))
        self.envs_done = 0
        cols = ['com_x1', 'com_y1', 'com_x2', 'com_y2'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)

        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
        self.plane_size = np.array([(0 - 0.063, 0 - 0.2095), (0.2625 + 0.063, 0.303107 + 0.1865)]) # 1000*np.array([(0.13125-0.025, 0.1407285-0.055),(0.13125+0.025, 0.1407285+0.055)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.nn_helper = helper.NNHelper(self.plane_size, real_or_sim="sim")
        # self.save_iters = 0
        
        """ Fingertip Vars """
        self.finger_positions = np.zeros((8,8)).tolist()
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if i%2!=0:
                    self.finger_positions[i][j] = gymapi.Vec3(i*0.0375, j*0.043301 - 0.02165, 1.5)
                else:
                    self.finger_positions[i][j] = gymapi.Vec3(i*0.0375, j*0.043301, 1.5)
        self.neighborhood_fingers = [[] for _ in range(self.scene.n_envs)]
        self.attraction_error = np.zeros((8,8))
        self.finga_q = gymapi.Quat(0, 0.707, 0, 0.707)
        self.active_idxs = {}
        self.active_IDs = set()
        
        self.KMeans = KMeans(n_clusters=64, random_state=69, n_init='auto')

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.obj_attr_handles = {}
        self.time_horizon = 155 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 10000001
        # self.current_episode = -self.scene.n_envs
        self.current_episode = 0
        self.dont_skip_episode = True

        """ Visual Servoing and RL Vars """
        self.bd_pts_dict = pkl.load(open('./config/assets/obj_props.pkl', 'rb'))

        self.bd_pts = {}
        self.goal_bd_pts = {}
        self.current_scene_frame = {}
        self.batch_size = hp_dict['batch_size']
        self.exploration_cutoff = hp_dict['exploration_cutoff']
        
        self.model = img_embed_model
        self.transform = transform
        if len(agents) == 1:
            self.agent = agents[0]
        else:
            self.pretrained_agent = agents[0]
            self.agent = agents[1]

        self.goal_yaw_deg = np.zeros((self.scene.n_envs))
        self.n_idxs = np.zeros((self.scene.n_envs),dtype=np.int8)
        self.init_pose = np.zeros((self.scene.n_envs, 4))
        self.goal_pose = np.zeros((self.scene.n_envs, 3))
        self.state_dim = hp_dict['state_dim']
        self.init_state = np.zeros((self.scene.n_envs, self.max_agents, self.state_dim))
        self.init_grasp_state = np.zeros((self.scene.n_envs, self.max_agents, 4))
        self.final_state = np.zeros((self.scene.n_envs, self.max_agents, self.state_dim))
        self.nn_bd_pts = {}
        self.actions_grasp = np.zeros((self.scene.n_envs, self.max_agents, 2))
        self.act_grasp_pix = np.zeros((self.scene.n_envs, self.max_agents, 2))
        self.actions = np.zeros((self.scene.n_envs, self.max_agents, 2))
        # self.actions_rb = np.zeros((self.scene.n_envs, self.max_agents, 2))
        self.pos = np.zeros((self.scene.n_envs, self.max_agents, 1))

        # self.ep_rewards = []
        self.ep_reward = np.zeros(self.scene.n_envs)
        self.ep_len = np.zeros(self.scene.n_envs)

        self.start_time = time.time()

        self.temp_cutoff_1 = 200
        self.temp_cutoff_2 = 2*self.temp_cutoff_1
        self.vs_rews = []
        self.rand_rews = []

        """ Diffusion Policy Utils """
        self.vis_servo_data = {}
        for env_idx in range(self.scene.n_envs):
            self.vis_servo_data[env_idx] = {'state': [], 'action': [], 'next_state':[], 'reward': [], 'obj_name': []}
        
        names = ['block', 'disc', 'hexagon', 'parallelogram', 'semicircle', 'shuriken', 'star', 'trapezium', 'triangle']
        encoder = LabelEncoder()
        # self.obj_name_encoder = encoder.fit_transform(np.array(self.obj_names).ravel())

        with open('./utils/MADP/normalizer.pkl', 'rb') as f:
            normalizer = pkl.load(f)
        self.state_scaler = normalizer['state_scaler']
        self.action_scaler = normalizer['action_scaler']
        self.obj_name_encoder = normalizer['obj_name_encoder']

        """ Test Traj Utils """
        self.test_trajs = {}
        self.current_traj = []
        self.current_traj_id = 0
        self.new_traj_bool = False
        self.init_traj_pose = None
        self.goal_traj_pose = None
        self.MegaTestingLoop = [pkl.load(open('./data/test_trajs.pkl', 'rb'))]*len(self.obj_names)

        self.tracked_trajs = {}
        for name in self.obj_names:
            self.tracked_trajs[name] = {'traj': [], 'error': []}
        
        """ Debugging and Visualization Vars """
        self.video_frames = np.zeros((self.hp_dict['inference_length'], (self.time_horizon - 4)*2, 480, 640, 3))

        # Traj testing code for GFT debugging
        # self.og_gft = None
        # self.new_gft = []
        # corners = np.array([[0, 0], [0.2625, 0], [0.2625, 0.2414], [0, 0.2414], [0, 0]])
        # num_steps = 77
        # steps_per_side = num_steps // 4
        # self.traj = []
        # for i in range(4):
        #     side_points = np.linspace(corners[i], corners[i+1], steps_per_side, endpoint=False)
        #     self.traj.extend(side_points)
        # self.traj.append([0, 0])
        # self.traj = np.array(self.traj[:num_steps])

        self.temp_var = {'initial_l2_dist': [],
                       'angle_diff': [],    
                        'reward': [],
                        'z_dist': []}
        # self.bd_pts_dict = {}

    def set_attractor_handles(self, env_idx):
        """ Creates an attractor handle for each fingertip """
        env_ptr = self.scene.env_ptrs[env_idx]
        self.attractor_handles[env_ptr] = np.zeros((8,8)).tolist()
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                attractor_props = gymapi.AttractorProperties()
                attractor_props.stiffness = self.cfg['capsule']['attractor_props']['stiffness']
                attractor_props.damping = self.cfg['capsule']['attractor_props']['damping']
                attractor_props.axes = gymapi.AXIS_ALL

                attractor_props.rigid_handle = self.scene.gym.get_rigid_handle(env_ptr, f'fingertip_{i}_{j}', 'capsule')
                self.attractor_handles[env_ptr][i][j] = self.scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)

        # self.obj_attr_handles[env_ptr] = {}
        # for obj_name in self.obj_names:
        #     attractor_props = gymapi.AttractorProperties()
        #     attractor_props.stiffness = self.cfg['capsule']['attractor_props']['stiffness']
        #     attractor_props.damping = self.cfg['capsule']['attractor_props']['damping']
        #     attractor_props.axes = gymapi.AXIS_ALL

        #     attractor_props.rigid_handle = self.scene.gym.get_rigid_handle(env_ptr, obj_name, 'block')
        #     self.obj_attr_handles[env_ptr][obj_name] = self.scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)

    def add_asset(self, scene):
        """ helper function to set up scene """
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                scene.add_asset(f'fingertip_{i}_{j}', self.fingertips[i][j], gymapi.Transform())

    def set_all_fingers_pose(self, env_idx, pos_high = True, all_or_one = "all"):
        """ Set the fingers to high/low pose """
        env_ptr = self.scene.env_ptrs[env_idx]
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if pos_high:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)])
                elif (i,j) in self.neighborhood_fingers[env_idx][1]:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q)])

    def set_nn_fingers_pose_low(self, env_idx, idxs):
        """ Set the fingers in idxs to low pose """
        for (i,j) in idxs:
            self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.49), r=self.finga_q)])

    def convert_world_2_pix(self, vec):
        if vec.shape[0] == 2:
            return (vec[0] - self.plane_size[0][0])/(self.delta_plane_x)*1080, 1920 - (vec[1]  - self.plane_size[0][1])/(self.delta_plane_y)*1920
        else:
            vec = vec.flatten()
            return (vec[0] - self.plane_size[0][0])/(self.delta_plane_x)*1080, 1920 - (vec[1]  - self.plane_size[0][1])/(self.delta_plane_y)*1920, vec[2]
    
    def scale_world_2_pix(self, vec):
        if isinstance(vec, np.ndarray):
            return vec / self.delta_plane * self.img_size
        else:
            return vec[0]/(self.delta_plane_x)*1080, -vec[1]/(self.delta_plane_y)*1920

    def convert_pix_2_world(self, vec):
        if vec.shape[0] == 2:
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], (1920 - vec[1])/1920*self.delta_plane_y + self.plane_size[0][1]
        else:
            vec = vec.flatten()
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], (1920 - vec[1])/1920*self.delta_plane_y + self.plane_size[0][1], vec[2]

    def scale_pix_2_world(self, vec):
        if isinstance(vec, np.ndarray):
            return vec / self.img_size * self.delta_plane
        else:
            return vec[0]/1080*self.delta_plane_x, -vec[1]/1920*self.delta_plane_y

    def set_block_pose(self, env_idx, goal=False):
        """ 
            Set the block pose to a random pose
            Store 2d goal pose in final and init state and 2d init pose in init state
            State dimensions are scaled by normalized pixel distance values. i.e state (mm) -> pixel / (1920,1080)
        """
        # T = [0.13125, 0.1407285]
        # 0.0, -0.02165, 0.2625, 0.303107

        if goal:
            self.object[env_idx].set_rb_transforms(env_idx, self.obj_name[env_idx], [gymapi.Transform(p=self.obj_dict[self.obj_name[env_idx]][1], r=self.obj_dict[self.obj_name[env_idx]][3])])
            # self.scene.gym.set_attractor_target(self.scene.env_ptrs[env_idx], self.obj_attr_handles[self.scene.env_ptrs[env_idx]][self.obj_name[env_idx]], gymapi.Transform(p=self.obj_dict[self.obj_name[env_idx]][2], r=self.obj_dict[self.obj_name[env_idx]][3]))
            self.obj_name[env_idx] = random.choice(self.obj_names)
            self.object[env_idx], object_p, _, object_r = self.obj_dict[self.obj_name[env_idx]]

            yaw = np.random.uniform(-np.pi, np.pi)
            r = R.from_euler('xyz', [0, 0, yaw])
            object_r = gymapi.Quat(*r.as_quat())
            T = (np.random.uniform(0.009, 0.19), np.random.uniform(0.007, 0.23))
            self.goal_pose[env_idx] = np.array([T[0], T[1], yaw])

            _, boundary_points, normals, initial_pose = self.bd_pts_dict[self.obj_name[env_idx]]
            tfed_bd_pts, _ = geom_utils.compute_transformation(boundary_points, normals, initial_pose, final_pose=(T[0], T[1], yaw))
            self.goal_bd_pts[env_idx] = tfed_bd_pts
        else:
            yaw = self.goal_pose[env_idx][2] + np.random.uniform(-0.78539, 0.78539)
            r = R.from_euler('xyz', [0, 0, yaw])
            object_r = gymapi.Quat(*r.as_quat())
            com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
            T = (com.p.x + np.random.uniform(-0.02, 0.02), com.p.y + np.random.uniform(-0.02, 0.02))
            self.init_pose[env_idx] = np.array([T[0], T[1], yaw, com.p.z])

        block_p = gymapi.Vec3(*T, 1.002)
        self.object[env_idx].set_rb_transforms(env_idx, self.obj_name[env_idx], [gymapi.Transform(p=block_p, r=object_r)])
    
    def ret_2d_pos(self, env_idx):
        com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
        quat = np.array((com.r.x, com.r.y, com.r.z, com.r.w))
        if (np.isnan(quat).any()):
            self.dont_skip_episode = False
            return None    
        _, _, yaw = R.from_quat([*quat]).as_euler('xyz')
        return com.p.x, com.p.y, yaw

    def set_traj_pose(self, env_idx, goal=False):
        return_exit = False
        if goal:
            if len(self.obj_names) == 0:
                return_exit = True

            if len(self.test_trajs) == 0:
                return_exit = True
                self.obj_name[env_idx] = self.obj_names.pop(0)
                self.object[env_idx], object_p, _, object_r = self.obj_dict[self.obj_name[env_idx]]
                self.test_trajs = self.MegaTestingLoop.pop(0)

            if len(self.current_traj) == 0:
                self.new_traj_bool = True
                traj_key = random.choice(list(self.test_trajs.keys()))
                self.current_traj = self.test_trajs.pop(traj_key)
                
                plt.plot(self.current_traj[:, 0], self.current_traj[:, 1], 'o', label=f'Curve Spline')
                plt.quiver(self.current_traj[:, 0], self.current_traj[:, 1], np.cos(self.current_traj[:, 2]), np.sin(self.current_traj[:, 2]))
                plt.show()
                self.current_traj = self.current_traj.tolist()
                self.init_traj_pose = self.current_traj.pop(0)

            self.goal_traj_pose = self.current_traj.pop(0)
            
            yaw = self.goal_traj_pose[2] + np.pi/2
            r = R.from_euler('xyz', [0, 0, yaw])
            object_r = gymapi.Quat(*r.as_quat())
            T = self.goal_traj_pose[:2]
            self.goal_pose[env_idx] = np.array([T[0], T[1], yaw])

            _, boundary_points, normals, initial_pose = self.bd_pts_dict[self.obj_name[env_idx]]
            tfed_bd_pts, _ = geom_utils.compute_transformation(boundary_points, normals, initial_pose, final_pose=(T[0], T[1], yaw))
            self.goal_bd_pts[env_idx] = tfed_bd_pts

            return return_exit, self.goal_pose[env_idx]
        else:
            com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
            quat = np.array((com.r.x, com.r.y, com.r.z, com.r.w))
            if (np.isnan(quat).any()):
                self.dont_skip_episode = False
                return None    
            _, _, yaw0 = R.from_quat([*quat]).as_euler('xyz')
            if self.new_traj_bool:
                self.new_traj_bool = False
                yaw = self.init_traj_pose[2] + np.pi/2
                r = R.from_euler('xyz', [0, 0, yaw])
                object_r = gymapi.Quat(*r.as_quat())
                T = self.init_traj_pose[:2]
                self.init_pose[env_idx] = np.array([T[0], T[1], yaw, com.p.z])

                block_p = gymapi.Vec3(*T, 1.002)
                self.object[env_idx].set_rb_transforms(env_idx, self.obj_name[env_idx], [gymapi.Transform(p=block_p, r=object_r)])
            else:
                self.init_pose[env_idx] = np.array([com.p.x, com.p.y, yaw0, com.p.z])
            return return_exit, (com.p.x, com.p.y, yaw0)

    def get_scene_image(self, env_idx):
        """ Render a camera image """
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        # self.current_scene_frame[env_idx] = frames['color'].data.astype(np.uint8)
        bgr_image = cv2.cvtColor(frames['color'].data.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_image, (640, 480), interpolation=cv2.INTER_AREA)

    def get_camera_image(self, env_idx):
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        return frames['color'].data.astype(np.uint8)

    def get_boundary_pts(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY)

        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        return seg_map, boundary_pts

    def get_nearest_robots_and_state(self, env_idx, final=False, init_bd_pts=None):
        """ 
        A helper function to get the nearest robot to the block and crop the image around it 
        Get camera image -> Segment it -> Convert boundary to cartesian coordinate space in mm ->
        Get nearest robot to the boundary -> Crop the image around the robot -> Resize to 224x224 ->
        Randomize the colors to get rgb image.

        Sets appropriate values for the following class variables:
        self.active_idxs
        self.actions
        self.init_state
        self.final_state
        self.bd_pts
        """
        img = self.get_camera_image(env_idx)
        seg_map, boundary_pts = self.get_boundary_pts(img)
        
        if len(boundary_pts) < 64:
            self.dont_skip_episode = False
            return None, None
        else:
            kmeans = self.KMeans.fit(boundary_pts)
            bd_cluster_centers = kmeans.cluster_centers_
            self.bd_pts[env_idx] = boundary_pts
            
            if not final:
                # Get indices of nearest robots to the boundary. We are not using neg_idxs for now. 
                idxs, neg_idxs = self.nn_helper.get_nn_robots(bd_cluster_centers)
                self.active_idxs[env_idx] = list(idxs)
                self.n_idxs[env_idx] = len(self.active_idxs[env_idx])
                
                self.actions[env_idx, :self.n_idxs[env_idx]] = np.array((0,0))
                self.actions_grasp[env_idx, :self.n_idxs[env_idx]] = np.array((0,0))
                
                _, self.nn_bd_pts[env_idx] = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs[env_idx], self.actions[env_idx])
                self.init_state[env_idx, :self.n_idxs[env_idx], :2] = [self.convert_pix_2_world(bd_pts) for bd_pts in self.nn_bd_pts[env_idx]]

                delta = self.goal_pose[env_idx] - self.init_pose[env_idx][:3]
                com = np.array(self.convert_pix_2_world(np.mean(boundary_pts, axis=0)))
                nn_bd_pts_world = [self.convert_pix_2_world(bd_pts) for bd_pts in self.nn_bd_pts[env_idx]]
                nn_bd_pts_world = geom_utils.transform_pts_wrt_com(nn_bd_pts_world, delta, com)
                
                self.init_state[env_idx, :self.n_idxs[env_idx], 2:4] = nn_bd_pts_world
                self.final_state[env_idx, :self.n_idxs[env_idx], 2:4] = nn_bd_pts_world

                self.init_grasp_state[env_idx, :self.n_idxs[env_idx], 2:4] = self.nn_helper.robot_positions[tuple(zip(*self.active_idxs[env_idx]))]/self.img_size
                self.init_grasp_state[env_idx, :self.n_idxs[env_idx], :2] = self.nn_bd_pts[env_idx]/self.img_size
                
                raw_rb_pos = self.nn_helper.rb_pos_raw[tuple(zip(*self.active_idxs[env_idx]))]

                if self.hp_dict['robot_frame']:
                    self.init_state[env_idx, :self.n_idxs[env_idx], :2] -= raw_rb_pos
                    self.init_state[env_idx, :self.n_idxs[env_idx], 2:4] -= raw_rb_pos
                    self.final_state[env_idx, :self.n_idxs[env_idx], 2:4] -= raw_rb_pos
                else:
                    self.init_state[env_idx, :self.n_idxs[env_idx], 4:6] = raw_rb_pos
                    self.final_state[env_idx, :self.n_idxs[env_idx], 4:6] = raw_rb_pos
                
            else:
                _, final_nn_bd_pts = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs[env_idx], self.actions[env_idx])
                self.final_state[env_idx, :self.n_idxs[env_idx], :2] = [self.convert_pix_2_world(bd_pts) for bd_pts in final_nn_bd_pts]
                if self.hp_dict['robot_frame']:
                    self.final_state[env_idx, :self.n_idxs[env_idx], :2] -= self.nn_helper.rb_pos_raw[tuple(zip(*self.active_idxs[env_idx]))]

                self.init_state[env_idx, :self.n_idxs[env_idx], 4:6] += self.actions_grasp[env_idx, :self.n_idxs[env_idx]]
                self.final_state[env_idx, :self.n_idxs[env_idx], 4:6] += self.actions[env_idx, :self.n_idxs[env_idx]]
                
################################################################################################################################################################################
                
                # if self.current_episode > 0:
                #     r_poses = self.nn_helper.rb_pos_raw[tuple(zip(*self.active_idxs[env_idx]))]
                #     init_pts = self.init_state[env_idx, :self.n_idxs[env_idx], :2].copy()
                #     init_bd_pts = np.array([self.convert_pix_2_world(bdpts) for bdpts in init_bd_pts])
                #     goal_bd_pts = self.init_state[env_idx, :self.n_idxs[env_idx], 2:4].copy()
                #     g_bd_pt2 = np.array([self.convert_pix_2_world(bdpts) for bdpts in self.goal_bd_pts[env_idx]])
                #     final_bd_pts = self.final_state[env_idx, :self.n_idxs[env_idx], :2].copy()
                #     act_grsp = self.actions_grasp[env_idx, :self.n_idxs[env_idx]].copy()
                #     acts = self.actions[env_idx, :self.n_idxs[env_idx]].copy()
                    
                #     # plt.figure(figsize=(10,17.78))
                #     plt.scatter(r_poses[:, 0], r_poses[:, 1], c='#880000ff')

                #     plt.scatter(g_bd_pt2[:, 0], g_bd_pt2[:, 1], c='#ffa50066')
                #     plt.scatter(init_bd_pts[:, 0], init_bd_pts[:, 1], c = '#00ff0066')
                #     plt.scatter(init_pts[:, 0], init_pts[:, 1], c = '#00ff00ff')
                #     plt.scatter(goal_bd_pts[:, 0], goal_bd_pts[:, 1], c='red')
                #     plt.scatter(final_bd_pts[:, 0], final_bd_pts[:, 1], c='blue')

                #     plt.quiver(r_poses[:, 0], r_poses[:, 1], act_grsp[:, 0], act_grsp[:, 1], scale=0.5, scale_units='xy')
                #     plt.quiver(init_pts[:, 0], init_pts[:, 1], acts[:, 0], acts[:, 1], scale=1, scale_units='xy')
                #     plt.gca().set_aspect('equal')
                #     plt.show()
                #     
    def get_nearest_robots_and_state_v2(self, env_idx, final=False, init_bd_pts=None):
        """ 
        A helper function to get the nearest robot to the block and crop the image around it 
        Get camera image -> Segment it -> Convert boundary to cartesian coordinate space in mm ->
        Get nearest robot to the boundary -> Crop the image around the robot -> Resize to 224x224 ->
        Randomize the colors to get rgb image.
        """
        # WE ARE NOT USING NORMALS in Expt#0. Might use later. 
        _, boundary_points, normals, initial_pose = self.bd_pts_dict[self.obj_name[env_idx]]

        com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
        quat = np.array((com.r.x, com.r.y, com.r.z, com.r.w))
        if (np.isnan(quat).any()):
            self.dont_skip_episode = False
            return None
            
        roll, pitch, yaw = R.from_quat([*quat]).as_euler('xyz')
        if (abs(roll) > 0.5)or(abs(pitch) > 0.5):
            self.dont_skip_episode = False
            return None
        else:
            final_pose = np.array((com.p.x, com.p.y, yaw))
            tfed_bd_pts, transformed_normals = geom_utils.compute_transformation(boundary_points, normals, initial_pose, final_pose)
            self.bd_pts[env_idx] = tfed_bd_pts            
            if not final:
                # Get indices of nearest robots to the boundary. We are not using neg_idxs for now. 
                idxs, neg_idxs = self.nn_helper.get_nn_robots_world(tfed_bd_pts)
                self.active_idxs[env_idx] = list(idxs)
                self.n_idxs[env_idx] = len(self.active_idxs[env_idx])
                
                self.actions[env_idx, :self.n_idxs[env_idx]] = np.array((0,0))
                self.actions_grasp[env_idx, :self.n_idxs[env_idx]] = np.array((0,0))
                
                _, self.nn_bd_pts[env_idx] = self.nn_helper.get_min_dist_world(tfed_bd_pts, self.active_idxs[env_idx], self.actions[env_idx])
                self.init_state[env_idx, :self.n_idxs[env_idx], :2] = self.nn_bd_pts[env_idx]

                delta = self.goal_pose[env_idx] - self.init_pose[env_idx][:3]
                com = np.mean(tfed_bd_pts, axis=0)
                tfed_nn_bd_pts = geom_utils.transform_pts_wrt_com(self.nn_bd_pts[env_idx], delta, com)
                
                self.init_state[env_idx, :self.n_idxs[env_idx], 2:4] = tfed_nn_bd_pts
                self.final_state[env_idx, :self.n_idxs[env_idx], 2:4] = tfed_nn_bd_pts

                self.init_grasp_state[env_idx, :self.n_idxs[env_idx], 2:4] = self.nn_helper.rb_pos_pix[tuple(zip(*self.active_idxs[env_idx]))]/self.img_size
                nn_bd_pts_pix = [self.convert_world_2_pix(bd_pts) for bd_pts in self.nn_bd_pts[env_idx]]
                self.init_grasp_state[env_idx, :self.n_idxs[env_idx], :2] = nn_bd_pts_pix/self.img_size
                
                raw_rb_pos = self.nn_helper.rb_pos_world[tuple(zip(*self.active_idxs[env_idx]))]

                if self.hp_dict['robot_frame']:
                    self.init_state[env_idx, :self.n_idxs[env_idx], :2] -= raw_rb_pos
                    self.init_state[env_idx, :self.n_idxs[env_idx], 2:4] -= raw_rb_pos
                    self.final_state[env_idx, :self.n_idxs[env_idx], 2:4] -= raw_rb_pos
                else:
                    self.init_state[env_idx, :self.n_idxs[env_idx], 4:6] = raw_rb_pos
                    self.final_state[env_idx, :self.n_idxs[env_idx], 4:6] = raw_rb_pos
                
            else:
                # print(f'{self.obj_name[env_idx]} final yaw: {yaw}')
                _, final_nn_bd_pts = self.nn_helper.get_min_dist_world(tfed_bd_pts, self.active_idxs[env_idx], self.actions[env_idx])
                self.final_state[env_idx, :self.n_idxs[env_idx], :2] = final_nn_bd_pts
                if self.hp_dict['robot_frame']:
                    self.final_state[env_idx, :self.n_idxs[env_idx], :2] -= self.nn_helper.rb_pos_world[tuple(zip(*self.active_idxs[env_idx]))]

                self.init_state[env_idx, :self.n_idxs[env_idx], 4:6] += self.actions_grasp[env_idx, :self.n_idxs[env_idx]]
                self.final_state[env_idx, :self.n_idxs[env_idx], 4:6] += self.actions[env_idx, :self.n_idxs[env_idx]]
                
################################################################################################################################################################################
                
                # if self.current_episode > 0:
                #     r_poses = self.nn_helper.rb_pos_world[tuple(zip(*self.active_idxs[env_idx]))]
                #     init_pts = self.init_state[env_idx, :self.n_idxs[env_idx], :2].copy()
                #     goal_bd_pts = self.init_state[env_idx, :self.n_idxs[env_idx], 2:4].copy()
                #     final_bd_pts = self.final_state[env_idx, :self.n_idxs[env_idx], :2].copy()
                #     act_grsp = self.actions_grasp[env_idx, :self.n_idxs[env_idx]].copy()
                #     acts = self.actions[env_idx, :self.n_idxs[env_idx]].copy()
                    
                #     plt.figure(figsize=(10,17.78))
                #     plt.scatter(r_poses[:, 0], r_poses[:, 1], c='#880000ff')

                #     plt.scatter(self.goal_bd_pts[env_idx][:, 0], self.goal_bd_pts[env_idx][:, 1], c='#ffa50066')
                #     plt.scatter(init_bd_pts[:, 0], init_bd_pts[:, 1], c = '#00ff0066')
                #     plt.scatter(init_pts[:, 0], init_pts[:, 1], c = '#00ff00ff')
                #     plt.scatter(goal_bd_pts[:, 0], goal_bd_pts[:, 1], c='red')
                #     plt.scatter(final_bd_pts[:, 0], final_bd_pts[:, 1], c='blue')

                #     plt.quiver(r_poses[:, 0], r_poses[:, 1], act_grsp[:, 0], act_grsp[:, 1], scale=0.5, scale_units='xy')
                #     plt.quiver(init_pts[:, 0], init_pts[:, 1], acts[:, 0], acts[:, 1], scale=1, scale_units='xy')
                #     plt.gca().set_aspect('equal')
                #     plt.show()
        return final_pose

    # def gaussian_reward_shaping(self, x, ampl, width, center=0, vertical_shift=0):
    #     return ampl * np.exp(-width * (x - center)**2) - vertical_shift

    # def reward_function(self, goal, final):
    #     loss = 5 * np.linalg.norm(goal - final)
    #     if loss < 1:
    #         loss = -0.5  * loss**2
    #     else:
    #         loss = -loss + 0.5
    #     return loss

    # def compute_reward_v2(self, env_idx, t_step):
    #     self.get_nearest_robots_and_state_v2(env_idx, final=True)
    #     self.ep_reward[env_idx] = self.reward_function(self.goal_bd_pts[env_idx], self.bd_pts[env_idx])
    
    def angle_difference(self, theta1, theta2):
        """ Calculate the shortest path difference between two angles """
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        return np.abs(delta_theta)
    
    def compute_reward(self, env_idx, t_step):
        """ 
        Computes the reward for the quasi-static phase
            1. Shortest distance between the robot and the block boundary
            2. ICP output transformation matrix to get delta_xy, and abs(theta)
        Combine both to compute the final reward
        """
        # This function is utterly incomplete. Fix it before running final init expt
        init_bd_pts = self.bd_pts[env_idx].copy()
        final_pose = self.get_nearest_robots_and_state_v2(env_idx, final=True, init_bd_pts=init_bd_pts)
        
        if final_pose is None:
            self.ep_reward[env_idx] = -5
            return
        
        if self.obj_name[env_idx]=="disc":
            self.ep_reward[env_idx] = 0
        else:
            self.ep_reward[env_idx] = -2*self.angle_difference(final_pose[2], self.goal_pose[env_idx, 2])
        loss = 100*np.linalg.norm(self.goal_pose[env_idx][:2] - final_pose[:2])
        if loss < 1:
            self.ep_reward[env_idx] -= 0.5  * loss**2
        else:
            self.ep_reward[env_idx] -= loss - 0.5
        
        return final_pose
        # com_world = np.mean(self.bd_pts[env_idx], axis=0)

        # rot = geom_utils.get_transform(self.goal_bd_pts[env_idx], self.bd_pts[env_idx])[2]
        # delta_com = np.mean(self.goal_bd_pts[env_idx], axis=0) - com_world
        # if self.obj_name[env_idx]=="disc":
        #     delta_2d_pose = np.array([*delta_com])
        # else:
        #     delta_2d_pose = np.array([*delta_com, rot])

        # # com = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
        # # yaw = R.from_quat([com.r.x, com.r.y, com.r.z, com.r.w]).as_euler('xyz')[2]
        
        # # rot_dif = abs(yaw - self.goal_pose[env_idx, 2])
        # # rot_dif = rot_dif if rot_dif < np.pi else 2*np.pi - rot_dif
        # # delta_2d_pose = [T[0] - self.goal_pose[env_idx, 0], T[1] - self.goal_pose[env_idx, 1], rot_dif]

        # # self.ep_reward[env_idx] = self.gaussian_reward_shaping(np.linalg.norm(delta_2d_pose), 15, 4500, -0.01, 5)
        # loss = 100*np.linalg.norm(delta_2d_pose)
        # loss2 = 100*np.linalg.norm(self.goal_bd_pts[env_idx] - self.bd_pts[env_idx])
        # if loss < 1:
        #     self.ep_reward[env_idx] = -0.5  * loss**2
        #     loss2 = -0.5  * loss2**2
        # else:
        #     self.ep_reward[env_idx] = -loss + 0.5
        #     loss2 = -loss2 + 0.5
        
        # print("Old Losss: ", self.ep_reward[env_idx])
        # print("New Loss: ", loss2)


        # if np.linalg.norm(delta_2d_pose) < 0.01:
        #     self.ep_reward[env_idx] = -np.mean((100*delta_2d_pose)**2)
        # else:
        #     self.ep_reward[env_idx] = -np.mean((100*delta_2d_pose)**2)


    def terminate(self, env_idx, t_step, agent):
        """ Update the replay buffer and reset the env """
        # Update policy
        self.compute_reward(env_idx, t_step)
        if (self.agent.ma_replay_buffer.size > self.batch_size) and (not self.hp_dict['vis_servo']):
            # epoch = self.scale_epoch(self.current_episode)
            # for i in range(5):
            self.agent.update(self.batch_size, self.current_episode)
        # else:
        #     print(f"Reward: {self.ep_reward[env_idx]} @ {self.current_episode}")

        if self.current_episode > self.scene.n_envs:
            self.log_data(env_idx, agent)
        # com = self.object.get_rb_transforms(env_idx, self.obj_name)[0]

        agent.ma_replay_buffer.store(self.init_state[env_idx], self.actions[env_idx], self.pos[env_idx], self.ep_reward[env_idx], self.final_state[env_idx], True, self.n_idxs[env_idx], self.obj_name[env_idx])
        if (self.current_episode%10000)==0:
            print(f"Reward: {self.ep_reward[env_idx]} @ {self.current_episode}")
            # agent.ma_replay_buffer.save_RB()

        # if self.current_episode == 500000:
        #     sys.exit(1)
        self.reset(env_idx)

    def reset(self, env_idx):
        """ Normal reset OR Alt-terminate state when block degenerately collides with robot. This is due to an artifact of the simulation. """
        self.table.set_rb_transforms(env_idx, 'table', [gymapi.Transform(p=gymapi.Vec3(0,0,0.5))])
        self.dont_skip_episode = True
        self.active_idxs[env_idx].clear()
        self.set_all_fingers_pose(env_idx, pos_high=True)
        self.ep_reward[env_idx] = 0
        self.init_state[env_idx] = np.zeros((self.max_agents, self.state_dim))
        self.final_state[env_idx] = np.zeros((self.max_agents, self.state_dim))
        self.actions[env_idx] = np.zeros((self.max_agents, 2))
        # self.actions_rb[env_idx] = np.zeros((self.max_agents, 2))
        self.pos[env_idx] = np.zeros((self.max_agents, 1))

    def env_step(self, env_idx, t_step, agent, test = False):
        if (self.ep_len[env_idx] == 0) and (t_step == 1):
            self.get_nearest_robots_and_state_v2(env_idx, final=False)
            self.set_nn_fingers_pose_low(env_idx, self.active_idxs[env_idx])

        if not self.dont_skip_episode:
            return

        if self.ep_len[env_idx] == 0:
            """ extract i, j, u, v for each robot """
            for i in range(len(self.active_idxs[env_idx])):
                self.actions_grasp[env_idx][i] = agent.get_actions(self.init_grasp_state[env_idx, i], deterministic=True) # For pretrained grasping policy, single state -> 2D action var

        elif (np.random.rand() <= self.hp_dict['epsilon']) or test:
            self.pos[env_idx, :self.n_idxs[env_idx], 0] = np.array([i[0]*8+i[1] for i in self.active_idxs[env_idx]]) #.reshape((-1,))
            # self.actions_rb[env_idx, :self.n_idxs[env_idx]] = agent.get_actions(self.init_state[env_idx, :self.n_idxs[env_idx]], self.pos[env_idx, :self.n_idxs[env_idx]], deterministic=test)
            # self.actions[env_idx, :self.n_idxs[env_idx]] = self.actions_grasp[env_idx][:self.n_idxs[env_idx]] + self.actions_rb[env_idx, :self.n_idxs[env_idx]]
            # self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(self.actions[env_idx, :self.n_idxs[env_idx]], -0.03, 0.03)
            self.actions[env_idx, :self.n_idxs[env_idx]] = agent.get_actions(self.init_state[env_idx, :self.n_idxs[env_idx]], self.pos[env_idx, :self.n_idxs[env_idx]], deterministic=test)
            self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(self.actions[env_idx, :self.n_idxs[env_idx]], -0.03, 0.03)
            # print(self.actions[env_idx, :self.n_idxs[env_idx]])
        else:
            # self.actions_rb[env_idx, :self.n_idxs[env_idx]] = np.random.uniform(-0.06, 0.06, size=(self.n_idxs[env_idx], 2))
            # self.actions[env_idx, :self.n_idxs[env_idx]] = self.actions_grasp[env_idx][:self.n_idxs[env_idx]] + self.actions_rb[env_idx, :self.n_idxs[env_idx]]
            # self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(self.actions[env_idx, :self.n_idxs[env_idx]], -0.03, 0.03)
            self.actions[env_idx, :self.n_idxs[env_idx]] = np.random.uniform(-0.03, 0.03, size=(self.n_idxs[env_idx], 2))

    def set_attractor_target(self, env_idx, t_step, actions, all_zeros=False):
        env_ptr = self.scene.env_ptrs[env_idx]
        if all_zeros:
            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q))
        else:
            for n, idx in enumerate(self.active_idxs[env_idx]):
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(actions[env_idx, n, 0], actions[env_idx, n, 1], -0.47), r=self.finga_q)) 
            
    def log_data(self, env_idx, agent):
        """ Store data about training progress in systematic data structures """
        if (not self.hp_dict["dont_log"]) and (agent.q_loss is not None):
            # wandb.log({"Delta Goal": np.linalg.norm(self.goal_pose[env_idx][:2] - self.init_pose[env_idx][:2])})
            wandb.log({"Reward":self.ep_reward[env_idx]})
            # wandb.log({"Q loss":np.clip(self.agent.q_loss, 0, 100)})

    def scale_epoch(self, x, A=100/np.log(100000), B=1/1000, C=1000):
        if x <= C:
            return 1
        else:
            return int(min(50, 1 + A * np.log(B * (x - C) + 1)))
    
    def inverse_dynamics(self, scene, env_idx, t_step, _):
        self.infer_iter = self.current_episode%self.hp_dict['infer_every']
        if self.infer_iter < self.hp_dict['inference_length']:
            self.test_learned_policy(scene, env_idx, t_step, _)
        else:
            t_step = t_step % self.time_horizon

            # if (t_step == 0) and (self.ep_len[env_idx] == 0):
            #     if self.current_episode < self.max_episodes:
            #     else:
            #         pass # Kill the pipeline somehow?
                
            if self.ep_len[env_idx]==0:
                if t_step == 0:
                    # img = self.get_camera_image(env_idx)
                    # _, self.goal_bd_pts[env_idx] = self.get_boundary_pts(img)
                    self.set_block_pose(env_idx) # Reset block to current initial pose
                    self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose
                    self.set_attractor_target(env_idx, t_step, None, all_zeros=True) # Set all fingers to high pose
                elif t_step == 1:
                    self.env_step(env_idx, t_step, self.pretrained_agent) #Store Init Pose
                elif (t_step == 2) and self.dont_skip_episode:
                    self.set_attractor_target(env_idx, t_step, self.actions_grasp)
                elif (t_step == self.time_horizon-1) and self.dont_skip_episode:
                    self.ep_len[env_idx] = 1
                elif not self.dont_skip_episode:
                    self.ep_len[env_idx] = 0
                    self.reset(env_idx)

            else:
                # Gen actions from new policy and set attractor until max episodes
                if t_step == 0:
                    if self.hp_dict["add_vs_data"] and (np.random.rand() <= self.hp_dict['ratio']):
                        self.vs_step(env_idx, t_step)
                    else:
                        self.env_step(env_idx, t_step, self.agent)
                elif t_step == 2:
                    self.set_attractor_target(env_idx, t_step, self.actions)
                elif t_step == (self.time_horizon-2):
                    self.current_episode += 1
                    self.terminate(env_idx, t_step, self.agent)
                elif t_step == self.time_horizon - 1:
                    self.set_block_pose(env_idx, goal=True) # Set block to next goal pose & Store Goal Pose for both states
                    self.ep_len[env_idx] = 0
            
    def test_learned_policy(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        
        if self.ep_len[env_idx]==0:
            if t_step == 0:
                # time.sleep(0.5)
                # img = self.get_camera_image(env_idx)
                # _, self.goal_bd_pts[env_idx] = self.get_boundary_pts(img)
                if self.hp_dict['test_traj']:
                    self.set_traj_pose(env_idx)
                else:
                    self.set_block_pose(env_idx) # Reset block to current initial pose
                self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose
                self.set_attractor_target(env_idx, t_step, None, all_zeros=True) # Set all fingers to high pose
            elif t_step == 1:
                self.env_step(env_idx, t_step, self.pretrained_agent) #Store Init Pose
            elif (t_step == 2) and self.dont_skip_episode:
                self.set_attractor_target(env_idx, t_step, self.actions_grasp)
            elif (t_step == self.time_horizon-1) and self.dont_skip_episode:
                self.ep_len[env_idx] = 1
            elif not self.dont_skip_episode:
                self.ep_len[env_idx] = 0
                self.reset(env_idx)
        else:         
            if t_step == 0:
                self.env_step(env_idx, t_step, self.agent, test=True) # Only Store Actions from MARL Policy
            elif t_step == 1:
                self.set_attractor_target(env_idx, t_step, self.actions)
            elif t_step == (self.time_horizon-2):
                init_bd_pts = self.bd_pts[env_idx].copy()
                final_pose = self.compute_reward(env_idx, t_step)
                # print(f"Reward: {self.ep_reward[env_idx]}")
                
                if final_pose is None:
                    return
                if not self.hp_dict["dont_log"]:
                    wandb.log({"Inference Reward":self.ep_reward[env_idx]})

                # if self.hp_dict['save_videos']:
                #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #     out = cv2.VideoWriter(f'./data/rl_data/{self.hp_dict["exp_name"]}/videos/{time.ctime()}.mp4', fourcc, 100, (640, 480))
                #     for image in self.video_frames[self.infer_iter]:
                #         out.write(image.astype(np.uint8))
                #     out.release()
                
                if self.hp_dict["print_summary"]:
                    
                    com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
                    self.temp_var['z_dist'].append(np.linalg.norm(self.init_pose[env_idx][3] - com.p.z))
                    # self.temp_var['initial_l2_dist'].append(np.linalg.norm(self.goal_bd_pts[env_idx] - init_bd_pts))
                    self.temp_var['initial_l2_dist'].append(np.linalg.norm(self.goal_pose[env_idx][:2] - self.init_pose[env_idx][:2]))
                    self.temp_var['angle_diff'].append(self.angle_difference(final_pose[2], self.goal_pose[env_idx, 2]))
                    self.temp_var['reward'].append(self.ep_reward[env_idx])
                    if self.current_episode%100 == 0:
                        with open(f"./init_vs_reward_diff_policy.pkl", "wb") as f:
                            pkl.dump(self.temp_var, f)
                self.reset(env_idx)
                self.current_episode += 1
            elif t_step == self.time_horizon - 1:
                if self.hp_dict['test_traj']:
                    self.set_traj_pose(env_idx, goal=True)
                else:
                    self.set_block_pose(env_idx, goal=True) 
                self.ep_len[env_idx] = 0

                # if self.current_episode > 50000:
                #     sys.exit(1)
            # else:
            #     self.video_frames[self.infer_iter, int((self.time_horizon-4)*self.ep_len[env_idx] + t_step-2)] = self.get_scene_image(env_idx)

    def diffusion_step(self, env_idx, agent):
            states = self.state_scaler.transform(self.init_state[env_idx, :self.n_idxs[env_idx]][None,...])
            states = torch.tensor(states, dtype=torch.float32)
            obj_name_enc = torch.tensor(self.obj_name_encoder.transform(np.array(self.obj_name[env_idx]).ravel())).to(torch.long)
            pos = torch.tensor(self.pos[env_idx, :self.n_idxs[env_idx]]).unsqueeze(0)
            # print(states.shape, obj_name_enc.shape, pos.shape)
            noise = torch.randn((1, self.n_idxs[env_idx], 2))
            print(noise.shape, states.shape, obj_name_enc.shape, pos.shape)
            denoised_actions = agent.actions_from_denoising_diffusion(noise, states, obj_name_enc, pos)
            actions = self.action_scaler.inverse_transform(denoised_actions.detach().cpu().numpy()).squeeze()
            print(actions)
            self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(denoised_actions, -0.03, 0.03)

    def test_diffusion_policy(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        
        if self.ep_len[env_idx]==0:
            if t_step == 0:
                # time.sleep(0.5)
                # img = self.get_camera_image(env_idx)
                # _, self.goal_bd_pts[env_idx] = self.get_boundary_pts(img)
                if self.hp_dict['test_traj']:
                    self.set_traj_pose(env_idx)
                else:
                    self.set_block_pose(env_idx) # Reset block to current initial pose
                self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose
                self.set_attractor_target(env_idx, t_step, None, all_zeros=True) # Set all fingers to high pose
            elif t_step == 1:
                self.env_step(env_idx, t_step, self.pretrained_agent) #Store Init Pose
            elif (t_step == 2) and self.dont_skip_episode:
                self.set_attractor_target(env_idx, t_step, self.actions_grasp)
            elif (t_step == self.time_horizon-1) and self.dont_skip_episode:
                self.ep_len[env_idx] = 1
            elif not self.dont_skip_episode:
                self.ep_len[env_idx] = 0
                self.reset(env_idx)
        else:         
            if t_step == 0:
                self.diffusion_step(env_idx, self.agent) # Only Store Actions from MARL Policy
                # self.vs_step(env_idx, t_step) # Only Store Actions from MARL Policy
            elif t_step == 1:
                self.set_attractor_target(env_idx, t_step, self.actions)
            elif t_step == (self.time_horizon-2):
                final_pose = self.compute_reward(env_idx, t_step)
                print(f"Reward: {self.ep_reward[env_idx]}")

                if self.hp_dict["print_summary"]:
                    
                    com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
                    self.temp_var['z_dist'].append(np.linalg.norm(self.init_pose[env_idx][3] - com.p.z))
                    # self.temp_var['initial_l2_dist'].append(np.linalg.norm(self.goal_bd_pts[env_idx] - init_bd_pts))
                    self.temp_var['initial_l2_dist'].append(np.linalg.norm(self.goal_pose[env_idx][:2] - self.init_pose[env_idx][:2]))
                    self.temp_var['angle_diff'].append(self.angle_difference(final_pose[2], self.goal_pose[env_idx, 2]))
                    self.temp_var['reward'].append(self.ep_reward[env_idx])
                    if self.current_episode%100 == 0:
                        with open(f"./init_vs_reward_diff_policy.pkl", "wb") as f:
                            pkl.dump(self.temp_var, f)
                self.reset(env_idx)
                self.current_episode += 1
            elif t_step == self.time_horizon - 1:
                if self.hp_dict['test_traj']:
                    exit_bool, pos = self.set_traj_pose(env_idx, goal=True)
                    self.tracked_trajs[self.obj_name[env_idx]]['traj'].append(pos)
                    self.tracked_trajs[self.obj_name[env_idx]]['error'].append((np.linalg.norm(self.goal_pose[env_idx][:2] - pos[:2]), self.angle_difference(pos[2], self.goal_pose[env_idx, 2])))
                else:
                    self.set_block_pose(env_idx, goal=True)
                
                self.ep_len[env_idx] = 0
                if exit_bool:
                    pkl.dump(self.tracked_trajs, open('./data/tracked_traj_data.pkl', 'wb'))
                    sys.exit(1)

    def vs_step(self, env_idx, t_step):
        
        _, boundary_points, normals, initial_pose = self.bd_pts_dict[self.obj_name[env_idx]]

        com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
        roll, pitch, yaw = R.from_quat([com.r.x, com.r.y, com.r.z, com.r.w]).as_euler('xyz')
        final_pose = (com.p.x, com.p.y, yaw)
        if (abs(roll) > 0.4)or(abs(pitch) > 0.4):
            self.dont_skip_episode = False
            return None, None
        else:
            tfed_bd_pts, transformed_normals = geom_utils.compute_transformation(boundary_points, normals, initial_pose, final_pose)
            self.bd_pts[env_idx] = tfed_bd_pts

            if self.current_episode >= 0:
                _, self.nn_bd_pts[env_idx] = self.nn_helper.get_min_dist_world(tfed_bd_pts, self.active_idxs[env_idx], self.actions_grasp[env_idx, :self.n_idxs[env_idx]])
                # com = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
                # yaw = R.from_quat([com.r.x, com.r.y, com.r.z, com.r.w]).as_euler('xyz')[2]

                # delta_com = self.goal_pose[env_idx] - np.array([com.p.x, com.p.y, yaw])
                # delta_com[:2] = self.scale_world_2_pix(delta_com[:2])
                com_world = np.mean(tfed_bd_pts, axis=0)

                rot = geom_utils.get_transform(self.goal_bd_pts[env_idx], tfed_bd_pts)[2]
                delta_com = np.mean(self.goal_bd_pts[env_idx], axis=0) - com_world

                tf_pts = self.nn_bd_pts[env_idx] - com_world
                rot_m = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                tf_pts = com_world + np.dot(tf_pts, rot_m)
                tf_pts += delta_com
                displacement_vectors = tf_pts - self.nn_bd_pts[env_idx]
                actions = self.actions_grasp[env_idx, :self.n_idxs[env_idx]] + displacement_vectors

                self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(actions, -0.03, 0.03)
                # self.actions[env_idx, :self.n_idxs[env_idx]] = np.random.uniform(-0.03, 0.03, size=(self.n_idxs[env_idx], 2))
                
                # print(self.obj_name[env_idx])
                # r_poses = self.nn_helper.rb_pos_world[tuple(zip(*self.active_idxs[env_idx]))]
                # init_pts = self.init_state[env_idx, :self.n_idxs[env_idx], :2].copy() + r_poses
                # goal_bd_pts = self.init_state[env_idx, :self.n_idxs[env_idx], 2:4].copy() + r_poses
                # act_grsp = self.actions_grasp[env_idx, :self.n_idxs[env_idx]].copy()
                # acts = self.actions[env_idx, :self.n_idxs[env_idx]].copy()
                # r_poses2 = r_poses + act_grsp
            
                # plt.scatter(r_poses[:, 0], r_poses[:, 1], c='#ddddddff')
                # plt.scatter(init_pts[:, 0], init_pts[:, 1], c = '#00ff00ff')
                # plt.scatter(goal_bd_pts[:, 0], goal_bd_pts[:, 1], c='red')

                # plt.quiver(r_poses[:, 0], r_poses[:, 1], act_grsp[:, 0], act_grsp[:, 1], color='#0000ff88')
                # plt.quiver(r_poses2[:, 0], r_poses2[:, 1], acts[:, 0], acts[:, 1], color='#ff000088')
                # # plt.quiver(r_poses2[:, 0], r_poses2[:, 1], act_gt[:, 0], act_gt[:, 1], color='#aaff5588')

                # plt.gca().set_aspect('equal')
                # plt.show()

    def vs_step_disc(self, env_idx, t_step):
        
        img = self.get_camera_image(env_idx)
        seg_map, new_bd_pts = self.get_boundary_pts(img)
        
        if len(new_bd_pts) < 64:
            self.dont_skip_episode = False
            return None, None
        else:
            if self.current_episode >= 0:
                kmeans = self.KMeans.fit(new_bd_pts)
                bd_cluster_centers = kmeans.cluster_centers_
                self.act_grasp_pix[env_idx, :self.n_idxs[env_idx]] = self.scale_world_2_pix(self.actions_grasp[env_idx, :self.n_idxs[env_idx]])
                _, self.nn_bd_pts[env_idx] = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs[env_idx], self.act_grasp_pix[env_idx])
    
                com_pix = np.mean(new_bd_pts, axis=0)

                # rot = 0
                delta_com = np.mean(self.goal_bd_pts[env_idx], axis=0) - com_pix

                # tf_pts = self.nn_bd_pts[env_idx] - com_pix
                # rot_m = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                # tf_pts = com_pix + np.dot(tf_pts, rot_m)
                # tf_pts += delta_com
                # displacement_vectors = tf_pts - self.nn_bd_pts[env_idx]
                tf_pts = self.nn_bd_pts[env_idx] + delta_com
                displacement_vectors = delta_com*np.ones((self.n_idxs[env_idx], 2))
                actions = self.act_grasp_pix[env_idx, :self.n_idxs[env_idx]] + displacement_vectors

                if self.hp_dict["add_vs_data"]:
                    # self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(self.scale_pix_2_world(actions), -0.03, 0.03)
                    self.actions[env_idx, :self.n_idxs[env_idx]] = np.clip(actions, -0.03, 0.03)
                else:
                    if self.current_episode < self.temp_cutoff_1:
                        # self.actions[env_idx, :self.n_idxs[env_idx]] = [self.scale_pix_2_world(disp_vec) for disp_vec in displacement_vectors]
                        self.actions[env_idx, :self.n_idxs[env_idx]] = displacement_vectors
                    elif self.current_episode < self.temp_cutoff_2:
                        act_rand = np.random.uniform(-0.06, 0.06, size=(self.n_idxs[env_idx], 2))
                        self.actions[env_idx, :self.n_idxs[env_idx]] = act_rand
                    else:
                        # Save vs_rews and rand_rews using pickle
                        pkl.dump(self.vs_rews, open(f"./data/manip_data/vis_servoing/vs_rews.pkl", "wb"))
                        pkl.dump(self.rand_rews, open(f"./data/manip_data/vis_servoing/rand_rews.pkl", "wb"))
                        sys.exit(1)
                
                
                
                # r_poses = self.nn_helper.robot_positions[tuple(zip(*self.active_idxs[env_idx]))]
                # # Plot robot positions of n_idxs
                # plt.scatter(r_poses[:, 0], r_poses[:, 1], c='#880000ff')

                # plt.scatter(new_bd_pts[:, 0], new_bd_pts[:, 1], c = 'green')
                # plt.scatter(self.goal_bd_pts[env_idx][:, 0], self.goal_bd_pts[env_idx][:, 1], c='orange')
                # plt.scatter(tf_pts[:, 0], tf_pts[:, 1], c='blue')
                # plt.scatter(self.nn_bd_pts[env_idx][:, 0], self.nn_bd_pts[env_idx][:, 1], c='#ff0000ff')
                # plt.quiver(self.nn_bd_pts[env_idx][:, 0], self.nn_bd_pts[env_idx][:, 1], displacement_vectors[:, 0], displacement_vectors[:, 1], scale=1, scale_units='xy')
                # plt.gca().set_aspect('equal')
                # plt.show()

    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        
        if self.ep_len[env_idx]==0:
            # Call the pretrained policy for all NN robots and set attractors
            if t_step == 0:
                # time.sleep(0.5)
                self.set_block_pose(env_idx) # Reset block to current initial pose
                self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose
                self.set_attractor_target(env_idx, t_step, None, all_zeros=True) # Set all fingers to high pose
            elif t_step == 1:
                self.env_step(env_idx, t_step, self.pretrained_agent) #Store Init Pose
            elif (t_step == 2) and self.dont_skip_episode:
                self.set_attractor_target(env_idx, t_step, self.actions_grasp)
            elif (t_step == self.time_horizon-1) and self.dont_skip_episode:
                self.ep_len[env_idx] = 1
            elif not self.dont_skip_episode:
                self.ep_len[env_idx] = 0
                self.reset(env_idx)
        
        else:
            # Gen actions from new policy and set attractor until max episodes            
            if t_step == 0:
                self.vs_step(env_idx, t_step) # Only Store Actions from MARL Policy
            elif t_step == 1:
                self.set_attractor_target(env_idx, t_step, self.actions)
            elif t_step == (self.time_horizon-3):
                self.compute_reward(env_idx, t_step)
                print(f"Reward: {self.ep_reward[env_idx]}")

                # if 0 < self.current_episode < self.temp_cutoff_1:
                #     self.vs_rews.append(self.ep_reward[env_idx])
                # elif self.temp_cutoff_1 <= self.current_episode < self.temp_cutoff_2:
                #     self.rand_rews.append(self.ep_reward[env_idx])
                # self.reset(env_idx)

                self.current_episode += 1
                self.pos[env_idx, :self.n_idxs[env_idx], 0] = np.array([i[0]*8+i[1] for i in self.active_idxs[env_idx]])
                self.terminate(env_idx, t_step, self.agent)
            elif t_step == (self.time_horizon - 2):
                self.set_block_pose(env_idx, goal=True) # Set block to next goal pose & Store Goal Pose for both states
            elif t_step == (self.time_horizon - 1):
                self.ep_len[env_idx] = 0

    def sim_test(self, scene, env_idx, t_step, _):
        """ Call this function to visualize the scene without taking any action """
        env_ptr = self.scene.env_ptrs[env_idx]
        t_step = t_step % self.time_horizon
        if t_step == 0:
            if self.og_gft is not None:
                self.og_gft.plot_embeddings(self.og_gft, self.new_gft)
            self.og_gft = None
            del self.new_gft
            self.new_gft = []
            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
        elif t_step == 1:
            img = self.get_camera_image(env_idx)
            seg_map, boundary_pts = self.get_boundary_pts(img)
            kmeans = self.KMeans.fit(boundary_pts)
            cluster_centers = kmeans.cluster_centers_
            pkl.dump(cluster_centers, open(f"cluster_centers.pkl", "wb"))
            self.og_gft = geom_utils.GFT(cluster_centers)
        elif t_step % 2 == 0:
            # 0.0, -0.02165, 0.2625, 0.303107
            # block_p = gymapi.Vec3(0 + 0.2625*t_step/self.time_horizon, 0.1407285, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
            xy = self.traj[(t_step-2)//2]
            block_p = gymapi.Vec3(*xy, 1.002)
            block_r = gymapi.Quat(0,0,0,1)
            self.object[env_idx].set_rb_transforms(env_idx, self.obj_name[env_idx], [gymapi.Transform(p=block_p, r=block_r)])
        else:
            img = self.get_camera_image(env_idx)
            seg_map, boundary_pts = self.get_boundary_pts(img)
            kmeans = self.KMeans.fit(boundary_pts)
            cluster_centers = kmeans.cluster_centers_
            self.new_gft.append(geom_utils.GFT(cluster_centers))

    def do_nothing(self, scene, env_idx, t_step, _):
        """ Call this function to visualize the scene without taking any action """
        env_ptr = self.scene.env_ptrs[env_idx]
        t_step = t_step % self.time_horizon
        
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if (i==0) and (j==0):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q))     
                else:
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
        if t_step==0:
            if len(self.obj_names) == 0:
                return_exit = True

            if len(self.test_trajs) == 0:
                return_exit = True
                self.obj_name[env_idx] = self.obj_names.pop(0)
                self.object[env_idx], object_p, _, object_r = self.obj_dict[self.obj_name[env_idx]]
                self.test_trajs = self.MegaTestingLoop.pop(0)

            if len(self.current_traj) == 0:
                self.new_traj_bool = True
                traj_key = random.choice(list(self.test_trajs.keys()))
                self.current_traj = self.test_trajs.pop(traj_key)
                
                plt.plot(self.current_traj[:, 0], self.current_traj[:, 1], 'o', label=f'Curve Spline')
                plt.quiver(self.current_traj[:, 0], self.current_traj[:, 1], np.cos(self.current_traj[:, 2]), np.sin(self.current_traj[:, 2]))
                plt.show()
                self.current_traj = self.current_traj.tolist()

            com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
            quat = np.array((com.r.x, com.r.y, com.r.z, com.r.w))
            if (np.isnan(quat).any()):
                self.dont_skip_episode = False
                return None    
            _, _, yaw0 = R.from_quat([*quat]).as_euler('xyz')
            self.init_traj_pose = self.current_traj.pop(0)
            yaw = self.init_traj_pose[2] + np.pi/2

            print(yaw0, yaw, self.angle_difference(yaw0, yaw))

            r = R.from_euler('xyz', [0, 0, yaw])
            object_r = gymapi.Quat(*r.as_quat())
            T = self.init_traj_pose[:2]
            self.init_pose[env_idx] = np.array([T[0], T[1], yaw, com.p.z])

            block_p = gymapi.Vec3(*T, 1.002)
            self.object[env_idx].set_rb_transforms(env_idx, self.obj_name[env_idx], [gymapi.Transform(p=block_p, r=object_r)])

        # elif t_step == (self.time_horizon - 2):
        #     self.set_block_pose(env_idx, goal=True)

        #     com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name[env_idx])[0]
        #     roll, pitch, yaw = R.from_quat([com.r.x, com.r.y, com.r.z, com.r.w]).as_euler('xyz')
        #     print(roll, pitch, yaw)
        # if t_step == 0:
        #     self.obj_name = self.obj_names.pop()
        #     self.object[env_idx], _, object_r = self.obj_dict[self.obj_name]
        #     block_p = gymapi.Vec3(0.13125, 0.1407285, 1.002)
        #     r = R.from_euler('xyz', [0, 0, 0], degrees=True)
        #     object_r = gymapi.Quat(*r.as_quat())
        #     self.object[env_idx].set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p, r=object_r)])
        # elif t_step == 1:
        #     img = self.get_camera_image(env_idx)
        #     _, bd_pts_pix = self.get_boundary_pts(img)
        #     bd_pts = np.array([self.convert_pix_2_world(bd_pt) for bd_pt in bd_pts_pix])
        #     idxs = np.arange(len(bd_pts))
        #     idxs = np.random.choice(idxs, 256, replace=False)
        #     vertex_normals = geom_utils.compute_vertex_normals(bd_pts)
        #     normals = geom_utils.ensure_consistent_normals(bd_pts, vertex_normals)

        #     com = self.object[env_idx].get_rb_transforms(env_idx, self.obj_name)[0]
        #     yaw = R.from_quat([com.r.x, com.r.y, com.r.z, com.r.w]).as_euler('xyz')[2]
        #     pose = np.array([com.p.x, com.p.y, yaw])

        #     print(self.obj_name, pose)
        #     self.bd_pts_dict[self.obj_name] = (bd_pts_pix[idxs], bd_pts[idxs], normals[idxs], pose)
        #     with open('./config/assets/obj_props.pkl', 'wb') as f:
        #         pkl.dump(self.bd_pts_dict, f)
        # elif t_step == 2:
        #     self.object[env_idx].set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=gymapi.Vec3(0.13125, 5.1407285, 1.002), r=gymapi.Quat(0,0,0,1))])