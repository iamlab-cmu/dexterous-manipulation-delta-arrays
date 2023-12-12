import numpy as np
import time
import pickle as pkl
import time
import random
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
from scipy.spatial.transform import Rotation as R
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
import utils.log_utils as log_utils
import utils.rl_utils as rl_utils
# import utils.SAC.sac as sac
# import utils.SAC.reinforce as reinforce
import utils.geometric_utils as geom_utils
device = torch.device("cuda:0")

class DeltaArraySim:
    def __init__(self, scene, cfg, obj, obj_name, img_embed_model, transform, agents, num_tips = [8,8], max_agents=15):
        print("Ye Init kitni baar bula raha hai?")
        """ Main Vars """
        self.scene = scene
        self.cfg = cfg
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        self.object = obj
        self.obj_name = obj_name
        self.max_agents = max_agents
        
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
        self.time_horizon = 155 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 10000001
        self.current_episode = 0
        self.dont_skip_episode = True

        """ Visual Servoing and RL Vars """
        self.bd_pts = {}
        self.current_scene_frame = {}
        self.batch_size = 128
        self.exploration_cutoff = 250
        
        self.model = img_embed_model
        self.transform = transform
        if len(agents) == 1:
            self.agent = agent[0]
        else:
            self.pretrained_agent = agents[0]
            self.agent = agents[1]

        self.init_state = np.zeros((self.scene.n_envs, self.max_agents, 10))
        self.final_state = np.zeros((self.scene.n_envs, self.max_agents, 10))
        self.actions = np.zeros((self.scene.n_envs, self.max_agents, 2))

        self.ep_rewards = []
        self.ep_reward = np.zeros(self.scene.n_envs)
        self.ep_len = 0

        self.optimal_reward = 30
        self.ep_since_optimal_reward = 0

        self.start_time = time.time()

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

    def set_block_pose(self, env_idx, goal=False):
        """ 
            Set the block pose to a random pose
            Store 2d goal pose in final and init state and 2d init pose in init state
            State dimensions are scaled by normalized pixel distance values. i.e state (mm) -> pixel / (1920,1080)
        """
        # T = [0.13125, 0.1407285]
        # 0.0, -0.02165, 0.2625, 0.303107
        r = R.from_euler('xyz', [90, 0, 0], degrees=True)
        object_r = gymapi.Quat(*r.as_quat())
        yaw = np.arctan2(2*(object_r.w*object_r.z + object_r.x*object_r.y), 1 - 2*(object_r.x**2 + object_r.y**2))

        if goal:
            T = (np.random.uniform(0.009, 0.21), np.random.uniform(0.005, 0.25))
            self.init_state[env_idx, :, 3:6] = np.array([(T[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0]), 1 - (T[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1]), yaw])
            self.final_state[env_idx, :, 3:6] = np.array([(T[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0]), 1 - (T[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1]), yaw])
        else:
            com = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
            # TODO: Change this to a random movement
            T = (com.p.x + 0.01, com.p.y + 0)
            self.init_state[env_idx, :, 0:3] = np.array([(T[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0]), 1 - (T[1]- self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1]), yaw])

        block_p = gymapi.Vec3(*T, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p, r=object_r)])

    def get_scene_image(self, env_idx):
        """ Render a camera image """
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        self.current_scene_frame[env_idx] = frames

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
            
            if final:
                # We are not using min_dists for now. Try expt to see if it helps. 
                min_dists, _ = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs[env_idx], self.actions[env_idx])
                obj_2d_tf = geom_utils.get_transform(init_bd_pts, self.bd_pts[env_idx])
                self.nn_bd_pts = geom_utils.transform_pts(self.nn_bd_pts, obj_2d_tf)
                self.final_state[env_idx, :self.n_idxs, 6:8] = self.nn_bd_pts/self.img_size

                # TODO: Add actions to robot positions. 
                self.init_state[env_idx, :self.n_idxs, 8:10] += self.actions[env_idx, :self.n_idxs]
                self.final_state[env_idx, :self.n_idxs, 8:10] += self.actions[env_idx, :self.n_idxs]
                return obj_2d_tf
            else:
                # Get indices of nearest robots to the boundary. We are not using neg_idxs for now. 
                idxs, neg_idxs = self.nn_helper.get_nn_robots(bd_cluster_centers)
                self.active_idxs[env_idx] = list(idxs)
                self.n_idxs = len(self.active_idxs[env_idx])
                # for n, idx in enumerate(self.active_idxs[env_idx]):
                self.actions[env_idx, :self.n_idxs] = np.array((0,0))
                
                _, self.nn_bd_pts = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs[env_idx], self.actions[env_idx])
                self.init_state[env_idx, :self.n_idxs, 6:8] = self.nn_bd_pts/self.img_size
                self.init_state[env_idx, :self.n_idxs, 8:10] = self.nn_helper.robot_positions[tuple(zip(*self.active_idxs[env_idx]))]/self.img_size
                self.final_state[env_idx, :self.n_idxs, 8:10] = self.nn_helper.robot_positions[tuple(zip(*self.active_idxs[env_idx]))]/self.img_size
            return True

    def compute_reward(self, env_idx, t_step):
        """ 
        Computes the reward for the quasi-static phase
            1. Shortest distance between the robot and the block boundary
            2. ICP output transformation matrix to get delta_xy, and abs(theta)
        Combine both to compute the final reward
        """
        # This function is utterly incomplete. Fix it before running final init expt
        init_bd_pts = self.bd_pts[env_idx]
        delta_2d_pose = self.get_nearest_robots_and_state(env_idx, final=True, init_bd_pts=init_bd_pts)
        
        print(delta_2d_pose)
        self.ep_reward[env_idx] += -np.linalg.norm(delta_2d_pose)
        # self.ep_reward[env_idx] += -tf_loss*0.6
        return True

    def terminate(self, env_idx, t_step, agent):
        """ Update the replay buffer and reset the env """
        # agent.replay_buffer.push(self.init_state[env_idx], self.action[env_idx], self.ep_reward[env_idx], self.final_state, True)
        if self.ep_reward[env_idx] > -180:
            if agent.ma_replay_buffer.size > self.batch_size:
                self.log_data(env_idx, t_step)

            #normalize the reward for easier training
            # self.ep_reward[env_idx] = (self.ep_reward[env_idx] - -45)/90
            agent.ma_replay_buffer.store(self.init_state[env_idx], self.actions[env_idx], self.ep_reward[env_idx], self.final_state, True)
            self.ep_rewards.append(self.ep_reward[env_idx])
            agent.logger.store(EpRet=self.ep_reward[env_idx], EpLen=self.ep_len)
            # if env_idx == (self.scene.n_envs-1):
            print(f"Iter: {self.current_episode}, Mean Reward: {np.mean(self.ep_rewards[-50:])}, Current Reward: {self.ep_reward[env_idx]}")
        self.reset(env_idx)

    def reset(self, env_idx):
        """ Normal reset OR Alt-terminate state when block degenerately collides with robot. This is due to an artifact of the simulation. """
        self.active_idxs[env_idx].clear()
        self.set_all_fingers_pose(env_idx, pos_high=True)
        self.ep_reward[env_idx] = 0
        self.init_state = np.zeros((self.scene.n_envs, self.max_agents, 10))
        self.final_state = np.zeros((self.scene.n_envs, self.max_agents, 10))
        self.actions = np.zeros((self.scene.n_envs, self.max_agents, 2))

    def env_step(self, env_idx, t_step, agent):
        if (self.ep_len == 0) and (t_step == 1):
            self.get_nearest_robots_and_state(env_idx, final=False)
            self.set_nn_fingers_pose_low(env_idx, self.active_idxs[env_idx])

        if not self.dont_skip_episode:
            return

        if self.ep_len == 0:
            """ extract i, j, u, v for each robot """
            for i in range(len(self.active_idxs[env_idx])):
                self.actions[env_idx][i] = agent.get_action(self.init_state[env_idx, i, 6:]) # For pretrained grasping policy, single state -> 2D action var
        elif (self.current_episode > self.exploration_cutoff):
            self.actions[env_idx, :self.n_idxs] = agent.get_actions(self.init_state[env_idx], self.n_idxs) # For MARL policy, multiple states -> 3D action var
        else:
            # self.actions[env_idx, :self.n_idxs] = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2))
            print("HAKUNA")
            self.actions[env_idx, :self.n_idxs] += np.ones((self.n_idxs, 2))*np.array((0.01, 0))
            self.actions[env_idx, :self.n_idxs] = np.clip(self.actions[env_idx, :self.n_idxs], -0.03, 0.03)

    def set_attractor_target(self, env_idx, t_step, all_zeros=False):
        env_ptr = self.scene.env_ptrs[env_idx]
        if all_zeros:
            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q))
        else:
            for n, idx in enumerate(self.active_idxs[env_idx]):
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(self.actions[env_idx, n, 0], self.actions[env_idx, n, 1], -0.47), r=self.finga_q)) 
                
                # Convert action to pix
                self.actions[env_idx, n, 0] = self.actions[env_idx, n, 0]/(self.plane_size[1][0]-self.plane_size[0][0])
                self.actions[env_idx, n, 1] = -1*self.actions[env_idx, n, 1]/(self.plane_size[1][1]-self.plane_size[0][1])
            

    def log_data(self, env_idx, t_step, agent):
        """ Store data about training progress in systematic data structures """
        log_utils.log_data(agent.logger, self.ep_rewards, self.current_episode, self.start_time)

    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon

        if (t_step == 0) and (self.ep_len == 0):
            if self.current_episode < self.max_episodes:
                self.current_episode += 1
            else:
                pass # Kill the pipeline somehow?
            
        if self.ep_len==0:
            # Call the pretrained policy for all NN robots and set attractors
            if t_step == 0:
                self.set_block_pose(env_idx) # Reset block to current initial pose
                self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose
                self.set_attractor_target(env_idx, t_step, all_zeros=True) # Set all fingers to high pose
            elif t_step == 1:
                self.env_step(env_idx, t_step, self.pretrained_agent) #Store Init Pose
            elif (t_step == 2) and self.dont_skip_episode:
                self.set_attractor_target(env_idx, t_step)
            elif (t_step == self.time_horizon-1) and self.dont_skip_episode:
                self.ep_len = 1
            elif not self.dont_skip_episode:
                self.ep_len = 0
                self.reset(env_idx)

        else:
            # Gen actions from new policy and set attractor until max episodes            
            if t_step == 0:
                self.env_step(env_idx, t_step, self.agent) # Only Store Actions from MARL Policy
            elif t_step == 2:
                self.set_attractor_target(env_idx, t_step)
            elif t_step == (self.time_horizon-2):
                # Update policy
                self.compute_reward(env_idx, t_step)
                if self.agent.ma_replay_buffer.size > self.batch_size:
                    self.agent.update(self.batch_size)
                self.terminate(env_idx, t_step, self.agent)
            elif t_step == self.time_horizon - 1:
                # Terminate episode
                self.set_block_pose(env_idx, goal=True) # Set block to next goal pose & Store Goal Pose for both states
                self.ep_len = 0
            
            
            
    def test_step(self, env_idx, t_step, env_ptr):
        """ 
        TODO: Update this whole thing

        Test_step has 3 functions
        1. Setup the env, get the initial state, query action from policy, and set_attractor_pose of the robot.
        2. Execute the action on the robot by set_attractor_target.
        3. set_attractor_pose of the robot to high pose so final image can be captured.
        """
        if t_step == 0:
            self.ep_reward[env_idx] = 0
            self.set_block_pose(env_idx) # Reset block to initial pose
            # self.get_scene_image(env_idx)
            self.set_all_fingers_pose(env_idx, pos_high=True) # Set all fingers to high pose

            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
            
            _, self.init_state[env_idx] = self.get_nearest_robots_and_state(env_idx, final=False)
            self.action[env_idx] = self.agent.test_policy(self.init_state[env_idx])
            self.set_nn_fingers_pose_low(env_idx, self.active_idxs[env_idx].keys())
        elif t_step == 1:
            for idx in self.active_idxs[env_idx].keys():
                self.active_idxs[env_idx][idx][0] = self.action[env_idx][0]/(self.plane_size[1][0]-self.plane_size[0][0])*1080
                self.active_idxs[env_idx][idx][1] = -1*self.action[env_idx][1]/(self.plane_size[1][1]-self.plane_size[0][1])*1920
                self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx[0]][idx[1]], gymapi.Transform(p=self.finger_positions[idx[0]][idx[1]] + gymapi.Vec3(self.action[env_idx][0], self.action[env_idx][1], -0.47), r=self.finga_q)) 
        elif t_step == self.time_horizon-2:
            self.set_all_fingers_pose(env_idx, pos_high=True)

    def test_learned_policy(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        
        if t_step in {0, 1, self.time_horizon-2}:
            self.test_step(env_idx, t_step, env_ptr)
        elif t_step == self.time_horizon-1:
            self.compute_reward(env_idx, t_step)
            print(f"Action: {self.action[env_idx]},Reward: {self.ep_reward[env_idx]}")

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
            block_p = gymapi.Vec3(*xy, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
            block_r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)
            self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p, r=block_r)])
        else:
            img = self.get_camera_image(env_idx)
            seg_map, boundary_pts = self.get_boundary_pts(img)
            kmeans = self.KMeans.fit(boundary_pts)
            cluster_centers = kmeans.cluster_centers_
            self.new_gft.append(geom_utils.GFT(cluster_centers))



