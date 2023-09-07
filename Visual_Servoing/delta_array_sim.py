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
        self.plane_size = 1000*np.array([(0.132-0.025, -0.179-0.055),(0.132+0.025, -0.179+0.055)])
        cols = ['com_x1', 'com_y1', 'com_x2', 'com_y2'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)
        
        # self.save_iters = 0
        # self.num_samples = 100
        
        """ Fingertip Vars """
        self.finger_positions = np.zeros((8,8)).tolist()
        self.kdtree_positions = np.zeros((64, 2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if i%2==0:
                    self.finger_positions[i][j] = gymapi.Vec3(i*0.0375, -j*0.043301 - 0.02165, 0.5)
                    self.kdtree_positions[i*8 + j, :] = (i*0.0375, -j*0.043301 - 0.02165)
                else:
                    self.finger_positions[i][j] = gymapi.Vec3(i*0.0375, -j*0.043301, 0.5)
                    self.kdtree_positions[i*8 + j, :] = (i*0.0375, -j*0.043301)
        self.neighborhood_fingers = [[] for _ in range(self.scene.n_envs)]
        self.contact_fingers = [set() for _ in range(self.scene.n_envs)]
        self.attraction_error = np.zeros((8,8))
        self.goal = np.zeros((8,8))
        self.finga_q = gymapi.Quat(0, 0.707, 0, 0.707)
        self.active_idxs = []

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.time_horizon = 152 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 1000
        self.current_episode = 0

        """ Visual Servoing and RL Vars """
        self.current_scene_frame = None
        self.batch_size = 32
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
        self.alpha_reward = -1
        self.psi_reward = 100
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
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)])
                elif (i,j) in self.neighborhood_fingers[env_idx][1]:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q)])

    def set_nn_fingers_pose(self, env_idx, idxs):
        for (i,j) in idxs:
            self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.49), r=self.finga_q)])

    def set_block_pose(self, env_idx):
        # block_p = gymapi.Vec3(np.random.uniform(0,0.313407), np.random.uniform(0,0.2803), self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        self.block_com[0] = np.array((0.132, -0.179))
        block_p = gymapi.Vec3(0.132, -0.179, self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p)])

    def get_scene_image(self, env_idx):
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        self.current_scene_frame = frames

    def get_nearest_robot_and_crop(self, env_idx):
        seg_map = self.current_scene_frame['seg'].data.astype(np.uint8)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        boundary_pts2 = np.array(np.where(boundary==255))
        min_x, min_y = np.min(boundary_pts, axis=0)
        max_x, max_y = np.max(boundary_pts, axis=0)
        # print(min_x, min_y, max_x, max_y)
        # print((max_x-min_x)*(self.plane_size[1][0]-self.plane_size[0][0]))
        boundary_pts[:,0] = (boundary_pts[:,0] - min_x)/(max_x-min_x)*(self.plane_size[1][0]-self.plane_size[0][0])+self.plane_size[0][0]
        boundary_pts[:,1] = (boundary_pts[:,1] - min_y)/(max_y-min_y)*(self.plane_size[1][1]-self.plane_size[0][1])+self.plane_size[0][1]

        ###################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@######################################
        """
        Debug The Weird Idx selection ISSUE
        """

        # transposed_points = boundary_pts.T
        # rotated_points = np.array([-transposed_points[1], transposed_points[0]]).T

        idxs, neg_idxs, DG, pos = self.nn_helper.get_nn_robots(boundary_pts, num_clusters=40)
        idxs = np.array(list(idxs))
        min_idx = tuple(idxs[np.lexsort((idxs[:, 1], idxs[:, 0]))][0])
        
        """ Single Robot Experiment. Change this to include entire neighborhood """
        self.active_idxs.append((0,0))
        self.active_idxs.append((1,0))
        # self.active_idxs = [tuple(idx) for idx in idxs]

        finger_pos = self.nn_helper.robot_positions[min_idx].copy()
        finger_pos[0] = (finger_pos[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0])*(max_x-min_x)+min_x
        finger_pos[1] = (finger_pos[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1])*(max_y-min_y)+min_y
        finger_pos = finger_pos.astype(np.int32)

        # plt.imshow(seg_map)
        # plt.scatter(finger_pos[1], finger_pos[0], c='r')
        # plt.show()

        crop = seg_map[finger_pos[0]-112:finger_pos[0]+112, finger_pos[1]-112:finger_pos[1]+112]
        cols = np.random.rand(3)
        crop = np.dstack((crop, crop, crop))*cols
        crop = Image.fromarray(np.uint8(crop*255))
        crop = self.transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            state = self.model(crop)
        return state.detach().cpu().squeeze()

    def compute_reward(self, env_idx, t_step):
        """ Computes reward, saves it in ep_reward variable and returns True if the action was successful """
        if t_step < self.time_horizon-2:
            self.ep_reward += self.alpha_reward
            return False
        else:
            final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
            self.block_com[1] = np.array((final_trans.p.x, final_trans.p.y))
            
            block_delta = np.linalg.norm(self.block_com[1] - self.block_com[0])
            if block_delta < 0.005:
                self.ep_reward += self.psi_reward
                return True
            else:
                self.ep_reward -= self.beta_reward * block_delta
                return False

    def terminate(self, env_idx, t_step):
        # Add episodes info in replay buffer
        final_state = self.get_nearest_robot_and_crop(env_idx)
        self.agent.replay_buffer.push(self.init_state, self.action, self.ep_reward, final_state, True)
        self.ep_rewards.append(self.ep_reward)
        print(self.ep_reward)

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
            print(self.action)
            print(f'Current Robot(s) Selected: {self.active_idxs}')
            self.set_nn_fingers_pose(env_idx, self.active_idxs)
        elif t_step == 1:
            for idx in self.active_idxs:
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(*self.action, -0.47), r=self.finga_q)) 
        elif t_step == 150:
            # Set the robots in idxs to default positions.
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        if t_step in {0, 1, self.time_horizon-2}:
            self.reset(env_idx, t_step, env_ptr)
        elif t_step < self.time_horizon-2:
            self.compute_reward(env_idx, t_step)
        elif t_step == self.time_horizon-1:
            self.compute_reward(env_idx, t_step)
            self.terminate(env_idx, t_step)