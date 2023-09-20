import numpy as np
import time
import pickle as pkl
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import networkx as nx
from PIL import Image
from scipy.spatial.transform import Rotation
np.set_printoptions(precision=4)
import wandb

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
import utils.SAC.reinforce as reinforce
from utils.geometric_utils import icp
device = torch.device("cuda:0")

# plt.ion()  # Turn on interactive mode

# fig, ax = plt.subplots()
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
        self.nn_helper = helper.NNHelper(self.plane_size)
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
        self.KMeans = KMeans(n_clusters=100, random_state=69, n_init=10)

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.time_horizon = 152 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 101
        self.current_episode = np.zeros(self.scene.n_envs)

        """ Visual Servoing and RL Vars """
        self.bd_pt_bool = True
        self.bd_pts = {}
        self.current_scene_frame = None
        self.batch_size = 4
        
        self.model = img_embed_model
        self.transform = transform
        self.agent = agent
        self.init_state = {}
        self.action = {}
        self.log_pi = {}

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
        """ helper function to set up scene """
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                scene.add_asset(f'fingertip_{i}_{j}', self.fingertips[i][j], gymapi.Transform())

    def set_all_fingers_pose(self, env_idx, pos_high = True, all_or_one = "all"):
        """ Set the fingers to high/low pose """
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if pos_high:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)])
                elif (i,j) in self.neighborhood_fingers[env_idx][1]:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q)])

    def set_nn_fingers_pose(self, env_idx, idxs):
        """ Set the fingers in idxs to low pose """
        for (i,j) in idxs:
            self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.49), r=self.finga_q)])

    def set_block_pose(self, env_idx):
        """ Set the block pose to a random pose """
        # T = [0.13125, 0.1407285]
        T = [0.11, 0.16]
        self.block_com[env_idx][0] = np.array((T))
        block_p = gymapi.Vec3(*T, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p)])

    def get_scene_image(self, env_idx):
        """ Render a camera image """
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        self.current_scene_frame = frames

    def get_nearest_robot_and_crop(self, env_idx, final=False):
        """ 
        A helper function to get the nearest robot to the block and crop the image around it 
        Get camera image -> Segment it -> Convert boundary to cartesian coordinate space in mm ->
        Get nearest robot to the boundary -> Crop the image around the robot -> Resize to 224x224 ->
        Randomize the colors to get rgb image, and Return resnet embedding of the crop. 
        """
        # plt.figure(figsize=(6.6667,11.85))
        img = self.current_scene_frame['color'].data.astype(np.uint8)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY)

        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        kmeans = self.KMeans.fit(boundary_pts)
        cluster_centers = kmeans.cluster_centers_
        
        self.bd_pts[env_idx] = cluster_centers
        idxs, neg_idxs = self.nn_helper.get_nn_robots(self.bd_pts[env_idx])
        idxs = np.array(list(idxs))
        min_idx = tuple(idxs[np.lexsort((idxs[:, 0], idxs[:, 1]))][0])
        
        """ Single Robot Experiment. Change this to include entire neighborhood """
        if not final:
            self.active_idxs[env_idx] = {min_idx: np.array((0,0))} # Store the action vector as value here later :)
        # [self.active_idxs[idx]=np.array((0,0)) for idx in idxs]

        # finger_pos = self.nn_helper.robot_positions[min_idx].astype(np.int32)
        # crop = seg_map[finger_pos[0]-112:finger_pos[0]+112, finger_pos[1]-112:finger_pos[1]+112]
        # # crop = cv2.resize(crop, (224,224))
        # cols = np.random.rand(3)
        # crop = np.dstack((crop, crop, crop))*cols
        # crop = Image.fromarray(np.uint8(crop*255))
        # crop = self.transform(crop).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     state = self.model(crop)
        # return state.detach().cpu().squeeze()

        min_dist, xy = self.nn_helper.get_min_dist(self.bd_pts[env_idx], self.active_idxs[env_idx])
        xy = torch.FloatTensor(xy)
        if self.bd_pt_bool:
            plt.imshow(seg_map)
            plt.scatter(xy[1], xy[0], c='r')
            plt.scatter(self.nn_helper.robot_positions[min_idx][1], self.nn_helper.robot_positions[min_idx][0], c='g')
            plt.scatter(self.bd_pts[env_idx][:,1], self.bd_pts[env_idx][:,0], c='b')
            plt.savefig(f"kmeans_img.png")
            self.bd_pt_bool = False
        # plt.show()
        return xy

    def reward_helper(self, env_idx, t_step):
        """ 
        Computes the reward for the quasi-static phase
        1. Shortest distance between the robot and the block boundary
        2. ICP output transformation matrix to get delta_xy, and abs(theta)
        Combine both to compute the final reward
        """
        init_bd_pts = self.bd_pts[env_idx]
        self.get_scene_image(env_idx)
        self.final_state = self.get_nearest_robot_and_crop(env_idx, final=True)

        M2 = icp(init_bd_pts, self.bd_pts[env_idx], icp_radius=1000)
        theta = np.arctan2(M2[1, 0], M2[0, 0])
        theta_degrees = np.rad2deg(theta)

        # final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
        # self.block_com[env_idx][1] = np.array((final_trans.p.x, final_trans.p.y))
        # block_l2_distance = np.linalg.norm(self.block_com[env_idx][1] - self.block_com[env_idx][0])
        tf = np.linalg.norm(M2[:2,3]) + abs(theta_degrees)
        nn_dist, xy = self.nn_helper.get_min_dist(self.bd_pts[env_idx], self.active_idxs[env_idx])
        return tf, nn_dist

    def compute_reward(self, env_idx, t_step):
        """ Computes reward, saves it in ep_reward variable and returns True if the action was successful """
        if t_step < self.time_horizon-2:
            # self.ep_reward += self.alpha_reward
            return False
        else:
            tf_loss, nn_dist_loss = self.reward_helper(env_idx, t_step)
            self.ep_reward[env_idx] += -nn_dist_loss[0]
            self.ep_reward[env_idx] += -tf_loss*0.6
            return True 

    def terminate(self, env_idx, t_step):
        """ Update the replay buffer and reset the env """
        # self.agent.replay_buffer.push(self.init_state[env_idx], self.action[env_idx], self.ep_reward[env_idx], self.final_state, True)
        self.agent.replay_buffer.push(self.init_state[env_idx], self.log_pi[env_idx], self.ep_reward[env_idx], self.final_state, True)
        self.ep_rewards.append(self.ep_reward[env_idx])
        # if env_idx == (self.scene.n_envs-1):
        print(f"Iter: {self.current_episode[env_idx]}, Reward: {self.ep_reward[env_idx]}")
        self.active_idxs[env_idx].clear()

    def reset(self, env_idx, t_step, env_ptr):
        """ 
        Reset has 3 functions
        1. Setup the env, get the initial state and action, and set_attractor_pose of the robot.
        2. Execute the action on the robot by set_attractor_target.
        3. set_attractor_pose of the robot to high pose so final image can be captured.
        """
        if t_step == 0:
            self.ep_reward[env_idx] = 0
            self.set_block_pose(env_idx) # Reset block to initial pose
            self.get_scene_image(env_idx)
            self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose

            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
            
            self.init_state[env_idx] = self.get_nearest_robot_and_crop(env_idx, final=False)
            self.action[env_idx], self.log_pi[env_idx] = self.agent.policy_nw.get_action(self.init_state[env_idx])
            self.action[env_idx] = self.agent.rescale_action(self.action[env_idx].detach().squeeze(0).numpy())
            self.set_nn_fingers_pose(env_idx, self.active_idxs[env_idx].keys())
        elif t_step == 1:
            for idx in self.active_idxs[env_idx].keys():
                self.active_idxs[env_idx][idx] = 1000*self.action[env_idx] # Convert action to mm
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(*self.action[env_idx], -0.47), r=self.finga_q)) 
        elif t_step == 150:
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def sim_test(self, env_idx, t_step, env_ptr):
        """ To visualize the scene without taking any action """
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.43), r=self.finga_q)) 

    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        # self.sim_test(env_idx, t_step, env_ptr)

        if (t_step == 0) and (env_idx == (self.scene.n_envs-1)):
            if self.current_episode[env_idx]%10 == 0:
                pkl.dump(self.ep_rewards, open('./data/rl_data/rl_data.pkl', 'wb'))
                self.agent.save_policy_model()
            if self.current_episode[env_idx] < self.max_episodes:
                self.current_episode[env_idx] += 1
            else:
                self.agent.end_wandb()
        if t_step in {0, 1, self.time_horizon-2}:
            self.reset(env_idx, t_step, env_ptr)
        elif t_step < self.time_horizon-2:
            pass # Compute quasi-static rewards
        elif t_step == self.time_horizon-1:
            self.compute_reward(env_idx, t_step)

            # if len(self.agent.replay_buffer) > self.batch_size:
            #     self.agent.update_policy(self.batch_size)
            self.agent.update_policy_reinforce(self.log_pi[env_idx], self.ep_reward[env_idx])

            self.terminate(env_idx, t_step)

    def test_step(self, env_idx, t_step, env_ptr):
        """ 
        Test_step has 3 functions
        1. Setup the env, get the initial state, query action from policy, and set_attractor_pose of the robot.
        2. Execute the action on the robot by set_attractor_target.
        3. set_attractor_pose of the robot to high pose so final image can be captured.
        """
        if t_step == 0:
            self.ep_reward[env_idx] = 0
            self.set_block_pose(env_idx) # Reset block to initial pose
            self.get_scene_image(env_idx)
            self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose

            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
            
            self.init_state[env_idx] = self.get_nearest_robot_and_crop(env_idx, final=False)
            self.action[env_idx] = self.agent.test_policy(self.init_state[env_idx])
            self.set_nn_fingers_pose(env_idx, self.active_idxs[env_idx].keys())
        elif t_step == 1:
            for idx in self.active_idxs[env_idx].keys():
                self.active_idxs[env_idx][idx] = 1000*self.action[env_idx] # Convert action to mm
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(*self.action[env_idx], -0.47), r=self.finga_q)) 
        elif t_step == 150:
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def test_learned_policy(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        
        if t_step in {0, 1, self.time_horizon-2}:
            self.test_step(env_idx, t_step, env_ptr)