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
import utils.SAC.sac as sac
device = torch.device("cuda:0")

class DeltaArraySim:
    def __init__(self, scene, cfg, obj, obj_name, num_tips = [8,8], run_no = 0):
        """ Main Vars """
        self.scene = scene
        self.cfg = cfg
        self.run_no = run_no
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        self.seed = 0
        self.image_id  = 0
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
        self.actions = {}
        self.pre_imgs = {}
        self.post_imgs = {}
        self.envs_done = 0
        self.block_com = np.array(((0.0,0.0),(0.0,0.0)))
        cols = ['com_x1', 'com_y1', 'com_x2', 'com_y2'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)

        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
        self.plane_size = 1000*np.array([(0 - 0.063, 0 - 0.2095), (0.2625 + 0.063, 0.303107 + 0.1865)]) # 1000*np.array([(0.13125-0.025, 0.1407285-0.055),(0.13125+0.025, 0.1407285+0.055)])
        # self.save_iters = 0
        # self.num_samples = 100
        
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
        self.current_episode = 0

        """ Visual Servoing and RL Vars """
        self.bd_pts = None
        self.current_scene_frame = None
        self.batch_size = 4
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model = self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 512}}
        self.agent = sac.SACAgent(env_dict=env_dict, 
            gamma=0.99, 
            tau=0.01, 
            alpha=0.2, 
            q_lr=3e-4, 
            policy_lr=3e-4,
            a_lr=3e-4, 
            buffer_maxlen=1000000
        )
        self.ep_rewards = []
        self.ep_reward = 0
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
        if pos_high:
            self.actions[env_idx] = np.zeros((8,8,2))
        elif all_or_one == "all":
            self.actions[env_idx] = np.random.uniform(-0.025, 0.025, (8,8,2))
        else:
            self.actions[env_idx] = np.zeros((8,8,2))

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
        self.block_com[0] = np.array((T))
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
        
        # plt.scatter(boundary_pts[:,0], boundary_pts[:,1], c='b')
        # plt.scatter(self.nn_helper.kdtree_positions[:,0], self.nn_helper.kdtree_positions[:,1], c='r')
        # plt.show()
        com = np.mean(boundary_pts, axis=0)

        if len(boundary_pts) > 200:
            self.bd_pts = boundary_pts[np.random.choice(range(len(boundary_pts)), size=200, replace=False)]
        else:
            self.bd_pts = boundary_pts
        idxs, neg_idxs = self.nn_helper.get_nn_robots(self.bd_pts)
        idxs = np.array(list(idxs))
        min_idx = tuple(idxs[np.lexsort((idxs[:, 0], idxs[:, 1]))][0])
        
        """ Single Robot Experiment. Change this to include entire neighborhood """
        self.active_idxs[min_idx] = np.array((0,0))
        # [self.active_idxs[idx]=np.array((0,0)) for idx in idxs]

        idxs = np.array(list(idxs))
        neighbors = self.nn_helper.robot_positions[idxs[:,0], idxs[:,1]]

        finger_pos = self.nn_helper.robot_positions[min_idx].copy()
        finger_pos[0] = (finger_pos[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0])*1080 - 0
        finger_pos[1] = 1920 - (finger_pos[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1])*1920
        finger_pos = finger_pos.astype(np.int32)

        # plt.imshow(seg_map)
        # for i in range(8):
        #     for j in range(8):
        #         finger_pos = self.nn_helper.robot_positions[i,j].copy()
        #         finger_pos[0] = (finger_pos[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0])*1080 - 0
        #         finger_pos[1] = 1920 - (finger_pos[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1])*1920
        #         finger_pos = finger_pos.astype(np.int32)

        #         plt.scatter(finger_pos[1], finger_pos[0], c='r')        
        # plt.show()

        crop = seg_map[finger_pos[0]-112:finger_pos[0]+112, finger_pos[1]-112:finger_pos[1]+112]
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
        final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
        self.block_com[1] = np.array((final_trans.p.x, final_trans.p.y))
        block_l2_distance = np.linalg.norm(self.block_com[1] - self.block_com[0])
    
        robot_to_obj_dists = self.nn_helper.get_min_dist(self.bd_pts, self.active_idxs)
        return block_l2_distance, robot_to_obj_dists

    def compute_reward(self, env_idx, t_step):
        """ Computes reward, saves it in ep_reward variable and returns True if the action was successful """
        if t_step < self.time_horizon-2:
            # self.ep_reward += self.alpha_reward
            return False
        else:
            block_l2_distance, robot_to_obj_dists = self.reward_helper(env_idx, t_step)
            # print(f'Rewards: {block_l2_distance, robot_to_obj_dists}')
            self.ep_reward += self.alpha_reward * robot_to_obj_dists[0]
            if 5e-4 < block_l2_distance < 0.005:
                self.ep_reward += self.psi_reward
                return True
            elif block_l2_distance < 5e-4:
                self.ep_reward += self.beta_reward * 100 * block_l2_distance
            else:
                self.ep_reward += self.beta_reward * block_l2_distance
                return False

    def terminate(self, env_idx, t_step):
        # Add episodes info in replay buffer
        final_state = self.get_nearest_robot_and_crop(env_idx)
        self.agent.replay_buffer.push(self.init_state, self.action, self.ep_reward, final_state, True)
        self.ep_rewards.append(self.ep_reward)
        print(f"Action: {self.action}, Reward: {self.ep_reward}")

    def reset(self, env_idx, t_step, env_ptr):
        if t_step == 0:
            self.ep_reward = 0
            self.active_idxs.clear()
            self.set_block_pose(env_idx) # Reset block to initial pose
            self.get_scene_image(env_idx) 
            self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose

            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
            
            self.init_state = self.get_nearest_robot_and_crop(env_idx)
            self.action = self.agent.get_action(self.init_state)
            # print(self.action)
            # print(f'Current Robot(s) Selected: {self.active_idxs.keys()}')
            self.set_nn_fingers_pose(env_idx, self.active_idxs.keys())
        elif t_step == 1:
            for idx in self.active_idxs.keys():
                self.active_idxs[idx] = self.action
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(*self.action, -0.47), r=self.finga_q)) 
        elif t_step == 150:
            # Set the robots in idxs to default positions.
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def sim_test(self, env_idx, t_step, env_ptr):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.43), r=self.finga_q)) 


    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
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