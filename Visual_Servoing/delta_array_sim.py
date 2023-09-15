import numpy as np
import time
import pickle as pkl
from scipy import spatial
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import networkx as nx
from PIL import Image
from scipy.spatial.transform import Rotation
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
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import utils.nn_helper as helper
import wandb
import utils.SAC.sac as sac
from utils.geometric_utils import icp
device = torch.device("cuda:0")

class DeltaArraySim:
    def __init__(self, scene, cfg, obj, obj_name, img_embed_model, transform, agent, num_tips = [8,8]):
        """ Main Vars """
        self.scene = scene
        self.cfg = cfg
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        self.object = obj
        self.obj_name = obj_name
        self.nn_helper = helper.NNHelper()
        
        # Introduce delta robot EE in the env
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.fingertips[i][j] = (GymCapsuleAsset(scene, **cfg['capsule']['dims'],
                    shape_props=cfg['capsule']['shape_props'],
                    rb_props=cfg['capsule']['rb_props'],
                    asset_options=cfg['capsule']['asset_options']
                ))
        """ Data Collection Vars """
        self.envs_done = 0
        self.block_com = np.zeros((self.scene.n_envs, 2, 2))
        cols = ['com_x1', 'com_y1', 'com_x2', 'com_y2'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)

        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
        self.plane_size = 1000*np.array([(0 - 0.063, 0 - 0.2095), (0.2625 + 0.063, 0.303107 + 0.1865)]) # 1000*np.array([(0.13125-0.025, 0.1407285-0.055),(0.13125+0.025, 0.1407285+0.055)])
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
        self.contact_fingers = [set() for _ in range(self.scene.n_envs)]
        self.attraction_error = np.zeros((8,8))
        self.goal = np.zeros((8,8))
        self.finga_q = gymapi.Quat(0, 0.707, 0, 0.707)
        self.active_idxs = {}

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.time_horizon = 152 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 1000
        self.current_episode = np.zeros(self.scene.n_envs)

        """ Visual Servoing and RL Vars """
        self.bd_pts = {}
        self.current_scene_frame = None
        self.batch_size = 64
        
        self.model = img_embed_model
        self.transform = transform
        self.agent = agent
        self.init_state = {}
        self.action = {}

        self.ep_rewards = []
        self.ep_reward = np.zeros(self.scene.n_envs)
        self.alpha_reward = -0.1
        self.psi_reward = 10
        self.beta_reward = -100

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

    def add_asset(self, scene):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                scene.add_asset(f'fingertip_{i}_{j}', self.fingertips[i][j], gymapi.Transform())

    def set_all_fingers_pose(self, env_idx, pos_high = True, all_or_one = "all"):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if pos_high:
                    if (i==0) and (j==0):
                        self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)])
                    else:
                        self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0.0, 0, 0), r=self.finga_q)])
                elif (i,j) in self.neighborhood_fingers[env_idx][1]:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q)])

    def set_nn_fingers_pose(self, env_idx, idxs):
        for (i,j) in idxs:
            self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.49), r=self.finga_q)])

    def set_block_pose(self, env_idx):
        # T = [0.13125, 0.1407285]
        T = [0.11, 0.16]
        self.block_com[env_idx][0] = np.array((T))
        block_p = gymapi.Vec3(*T, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p)])

    def get_scene_image(self, env_idx):
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        self.current_scene_frame = frames

    def get_nearest_robot_and_crop(self, env_idx):
        # plt.figure(figsize=(6.6667,11.85))
        img = self.current_scene_frame['color'].data.astype(np.uint8)
        # plt.imshow(img)
        # plt.show()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY)

        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        # 0, 1920 are origin pixels in img space
        boundary_pts[:,0] = (boundary_pts[:,0] - 0)/1080*(self.plane_size[1][0]-self.plane_size[0][0])+self.plane_size[0][0]
        boundary_pts[:,1] = (1920 - boundary_pts[:,1])/1920*(self.plane_size[1][1]-self.plane_size[0][1])+self.plane_size[0][1]
        if len(boundary_pts) > 200000:
            self.bd_pts[env_idx] = boundary_pts[np.random.choice(range(len(boundary_pts)), size=200, replace=False)]
        else:
            self.bd_pts[env_idx] = boundary_pts
        idxs, neg_idxs = self.nn_helper.get_nn_robots(self.bd_pts[env_idx])
        idxs = np.array(list(idxs))
        min_idx = tuple(idxs[np.lexsort((idxs[:, 0], idxs[:, 1]))][0])
        
        """ Single Robot Experiment. Change this to include entire neighborhood """
        self.active_idxs[env_idx] = {min_idx: np.array((0,0))} # Store the action vector as value here later :)
        # [self.active_idxs[idx]=np.array((0,0)) for idx in idxs]

        idxs = np.array(list(idxs))
        neighbors = self.nn_helper.robot_positions[idxs[:,0], idxs[:,1]]

        finger_pos = self.nn_helper.robot_positions[min_idx].copy()
        finger_pos[0] = (finger_pos[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0])*1080 - 0
        finger_pos[1] = 1920 - (finger_pos[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1])*1920
        finger_pos = finger_pos.astype(np.int32)

        crop = seg_map[finger_pos[0]-224:finger_pos[0]+224, finger_pos[1]-224:finger_pos[1]+224]
        crop = cv2.resize(crop, (224,224), interpolation=cv2.INTER_AREA)
        # plt.imshow(crop)
        # plt.show()
        cols = np.random.rand(3)
        crop = np.dstack((crop, crop, crop))*cols
        crop = Image.fromarray(np.uint8(crop*255))
        crop = self.transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            state = self.model(crop)
        return state.detach().cpu().squeeze()

    def reward_helper(self, env_idx, t_step):
        nn_dist = self.nn_helper.get_min_dist(self.bd_pts[env_idx], self.active_idxs[env_idx])

        init_bd_pts = self.bd_pts[env_idx]
        self.final_state = self.get_nearest_robot_and_crop(env_idx)
        M2 = icp(init_bd_pts, self.bd_pts[env_idx], icp_radius=1000)
        theta = np.arctan2(M2[1, 0], M2[0, 0])
        theta_degrees = np.rad2deg(theta)

        # final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
        # self.block_com[env_idx][1] = np.array((final_trans.p.x, final_trans.p.y))
        # block_l2_distance = np.linalg.norm(self.block_com[env_idx][1] - self.block_com[env_idx][0])
        tf = np.linalg.norm(M2[:2,3]) + abs(theta_degrees)
        print(M2)
        print(f"TF_loss: {tf}, nn_dist_loss: {nn_dist}")
        return tf, nn_dist

    def compute_reward(self, env_idx, t_step):
        """ Computes reward, saves it in ep_reward variable and returns True if the action was successful """
        if t_step < self.time_horizon-2:
            # self.ep_reward += self.alpha_reward
            return False
        else:
            tf_loss, nn_dist_loss = self.reward_helper(env_idx, t_step)
            self.ep_reward[env_idx] += nn_dist_loss[0]
            self.ep_reward[env_idx] += tf_loss
            return True 

    def terminate(self, env_idx, t_step):
        self.agent.replay_buffer.push(self.init_state[env_idx], self.action[env_idx], self.ep_reward[env_idx], self.final_state, True)
        self.ep_rewards.append(self.ep_reward[env_idx])
        print(f"Env_idx: {env_idx}, Iter: {self.current_episode[env_idx]}, Action: {self.action[env_idx]}, Reward: {self.ep_reward[env_idx]}")
        self.active_idxs[env_idx].clear()

    def reset(self, env_idx, t_step, env_ptr):
        if t_step == 0:
            self.ep_reward[env_idx] = 0
            self.set_block_pose(env_idx) # Reset block to initial pose
            self.get_scene_image(env_idx) 
            self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose

            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
            
            self.init_state[env_idx] = self.get_nearest_robot_and_crop(env_idx)
            self.action[env_idx] = self.agent.get_action(self.init_state[env_idx])
            self.set_nn_fingers_pose(env_idx, self.active_idxs[env_idx].keys())
        elif t_step == 1:
            for idx in self.active_idxs[env_idx].keys():
                self.active_idxs[env_idx][idx] = 1000*self.action[env_idx]
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(*self.action[env_idx], -0.47), r=self.finga_q)) 
        elif t_step == 150:
            # Set the robots in idxs to default positions.
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def sim_test(self, env_idx, t_step, env_ptr):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.43), r=self.finga_q)) 


    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        if (t_step == 0) and (env_idx == (self.scene.n_envs-1)):
            if self.current_episode[env_idx] < self.max_episodes:
                self.current_episode[env_idx] += 1
            else:
                self.agent.end_wandb()
        env_ptr = self.scene.env_ptrs[env_idx]
        # self.sim_test(env_idx, t_step, env_ptr)
        if t_step in {0, 1, self.time_horizon-2}:
            self.reset(env_idx, t_step, env_ptr)
        elif t_step < self.time_horizon-2:
            self.compute_reward(env_idx, t_step)
        elif t_step == self.time_horizon-1:
            self.compute_reward(env_idx, t_step)

            if len(self.agent.replay_buffer) > self.batch_size:
                self.agent.update(self.batch_size)
            self.terminate(env_idx, t_step)