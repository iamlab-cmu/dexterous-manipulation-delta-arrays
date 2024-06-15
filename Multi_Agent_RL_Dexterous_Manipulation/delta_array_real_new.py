import numpy as np
import time
import pickle as pkl
import time
import socket
import random
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
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
import threading

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

import utils.nn_helper as helper
import utils.geometric_utils as geom_utils

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

device = torch.device("cuda:0")
BUFFER_SIZE = 20

low_z = 12
high_z = 5.5

current_frame = None
lock = threading.Lock()

def capture_and_convert():
    global current_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Stream', frame)
        # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with lock:
            current_frame = frame
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

def start_capture_thread():
    capture_thread = threading.Thread(target=capture_and_convert)
    capture_thread.daemon = True
    capture_thread.start()


class DeltaArrayReal:
    def __init__(self, agents, objs, hp_dict, num_tips = [8,8], max_agents=64):
        """ Main Vars """
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.max_agents = max_agents
        self.hp_dict = hp_dict
        self.obj_dict = objs
        self.obj_names = list(self.obj_dict.keys())
        self.object = {}
        
        """ Data Collection Vars """
        self.lower_green_filter = np.array([30, 5, 5])
        self.upper_green_filter = np.array([90, 255, 255])
        self.img_size = np.array((1080, 1920))
        self.plane_size = np.array([(1.8, 1.9),(23.125, -36.225)])
        self.real2sim = np.array((100, -100))
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.nn_helper = helper.NNHelper(self.plane_size, real_or_sim="real")
        
        """ Fingertip Vars """
        # Real World Y-Axis is -Y as opposed to +Y in simulation
        self.finger_positions_cm = np.zeros((8,8,2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if i%2!=0:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301 + 2.165)
                else:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301)
        self.KMeans = KMeans(n_clusters=256, random_state=69, n_init='auto')

        """ Real World Util Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()
        self.neighborhood_fingers = []
        self.active_idxs = []
        self.active_IDs = set()

        """ Env Vars """
        self.state_dim = hp_dict['state_dim']

        self.goal_yaw_deg = 0
        self.init_pose = np.zeros((2))
        self.goal_bd_pts = None
        self.goal_pose = np.zeros((2))
        self.rot = 0

        self.init_state = np.zeros((self.max_agents, self.state_dim))
        self.init_grasp_state = np.zeros((self.max_agents, 4))
        self.final_state = np.zeros((self.max_agents, self.state_dim))
        self.pos = np.zeros((self.max_agents, 1))
        self.nn_bd_pts = {}

        self.actions_grasp = np.zeros((self.max_agents, 2))
        self.act_grasp_pix = np.zeros((self.max_agents, 2))
        self.actions = np.zeros((self.max_agents, 2))
        self.n_idxs = 0

        self.ep_reward = 0
        # self.actions = {}
        # self.actions_grasp = {}

        """ Camera Vars """
        # self.cam = cv2.VideoCapture(0)
        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        """ Visual Servoing and RL Vars """
        self.bd_pts = []
        self.current_scene_frame = []
        self.batch_size = 128
        self.exploration_cutoff = 250
        
        if len(agents) == 1:
            self.agent = agents[0]
        else:
            self.pretrained_agent = agents[0]
            self.agent = agents[1]

        """ Diffusion Policy Utils """
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

        self.obj_name = self.hp_dict['obj_name']
        self.tracked_trajs = {}
        self.tracked_trajs[self.obj_name] = {'traj': [], 'error': []}
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()

    def setup_delta_agents(self):
        self.delta_agents = {}
        # Obtain numbers of 2x2 grids
        for i in range(1, 17):
            # Get IP Addr and socket of each grid and classify them as useful or useless
            # if i!= 10:
            try:
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.1)
                self.delta_agents[i-1] = DeltaArrayAgent(esp01, i)
            except Exception as e:
                print("Error at robot ID: ", i)
                raise e
        self.reset()
        return
    
    def reset(self):
        self.active_idxs.clear()
        self.ep_reward = 0
        self.init_state = np.zeros((self.max_agents, self.state_dim))
        self.final_state = np.zeros((self.max_agents, self.state_dim))
        self.actions = np.zeros((self.max_agents, 2))
        self.pos = np.zeros((self.max_agents, 1))

        for i in set(self.RC.robo_dict_inv.values()):
            self.delta_agents[i-1].reset()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Resetting Delta Robots...")
        self.wait_until_done()
        print("Done!")
    
    def wait_until_done(self, topandbottom=False):
        done_moving = False
        while not done_moving:
            # print(self.to_be_moved)
            for i in self.to_be_moved:
                try:
                    received = i.esp01.recv(BUFFER_SIZE)
                    ret = received.decode().strip()
                    if ret == "A":
                        i.done_moving = True
                        time.sleep(0.1)
                except Exception as e: 
                    # print(e)
                    pass
            # print
            bool_dones = [i.done_moving for i in self.to_be_moved]
            # print(bool_dones)
            done_moving = all(bool_dones)
        time.sleep(0.1)
        for i in self.delta_agents:
            self.delta_agents[i].done_moving = False
        del self.to_be_moved[:]
        self.active_IDs.clear()
        # print("Done!")
        return
    
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

    def convert_real_2_sim(self, vec):
        if len(vec) == 2:
            return vec[0]/100, -vec[1]/100
        else:
            vec = vec.flatten()
            return vec[0]/100, -vec[1]/100, vec[2]
        
    def scale_pix_2_world(self, vec):
        if isinstance(vec, np.ndarray):
            return vec / self.img_size * self.delta_plane
        else:
            return vec[0]/1080*self.delta_plane_x, -vec[1]/1920*self.delta_plane_y
    
    def practicalize_traj(self, traj):
        traj[0] = [-0.3*traj[0][0], -0.3*traj[0][1], high_z]
        for i in range(1,5):
            traj[i] = [-0.3*traj[i][0], -0.3*traj[i][1], low_z]
        for i in range(5,20):
            traj[i] = [0.8*traj[i][0], 0.8*traj[i][1], low_z]
        return traj
    
    def practicalize_traj2(self, traj):
        for i in range(20):
            traj[i] = [0.8*traj[i][0], 0.8*traj[i][1], low_z]
        return traj
    
    def angle_difference(self, theta1, theta2):
        """ Calculate the shortest path difference between two angles """
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        return np.abs(delta_theta)
    
    def compute_reward(self):
        init_bd_pts = self.bd_pts.copy()
        self.get_nearest_robots_and_state(final=True, init_bd_pts=init_bd_pts)

        com_pix = np.mean(self.bd_pts, axis=0)
        rot = geom_utils.get_transform(self.goal_bd_pts, self.bd_pts)[2]
        delta_com = np.mean(self.goal_bd_pts, axis=0) - com_pix
        delta_com = self.scale_pix_2_world(delta_com)

        if self.obj_name=="disc":
            delta_2d_pose = np.array([*delta_com])
        else:
            delta_2d_pose = np.array([*delta_com, rot])

        loss = 100*np.linalg.norm(delta_2d_pose)
        if loss < 1:
            self.ep_reward = -0.5  * loss**2
        else:
            self.ep_reward = -loss + 0.5
    
    def get_seg_and_bd_pts(self):
        # ret, img = self.cam.read()
        # if not ret:
        #     return
        
        img = current_frame
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY)
        ret, seg_map = cv2.threshold(seg_map, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((11,11),np.uint8)
        seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_ERODE, kernel)
        seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_DILATE, kernel)
        seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel)
        seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        return boundary_pts, seg_map

    def get_nearest_robots_and_state(self,final=False):
        """ 
        A helper function to get the nearest robot to the block and crop the image around it 
        Get camera image -> Segment it -> Convert boundary to cartesian coordinate space in mm ->
        Get nearest robot to the boundary -> Crop the image around the robot -> Resize to 224x224 ->
        Randomize the colors to get rgb image.

        Returns nearest boundary distance and state of the MDP
        """
        boundary_pts, seg_map = self.get_seg_and_bd_pts()
        if len(boundary_pts) < 64:
            self.dont_skip_episode = False
            return None, None
        else:
            kmeans = self.KMeans.fit(boundary_pts)
            bd_cluster_centers = kmeans.cluster_centers_
            self.bd_pts = boundary_pts
            if not final:            
                idxs, neg_idxs = self.nn_helper.get_nn_robots(bd_cluster_centers)
                print(idxs)
                self.active_idxs = list(idxs)
                self.n_idxs = len(self.active_idxs)
                self.pos[:self.n_idxs, 0] = np.array([i[0]*8+i[1] for i in self.active_idxs]) 

                self.actions[:self.n_idxs] = np.array((0,0))
                self.actions_grasp[:self.n_idxs] = np.array((0,0))
                
                _, self.nn_bd_pts = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs, self.actions)
                self.init_state[:self.n_idxs, :2] = [self.convert_real_2_sim(self.convert_pix_2_world(bd_pts)) for bd_pts in self.nn_bd_pts.copy()]
                self.init_grasp_state[:self.n_idxs, 2:4] = self.nn_helper.rb_pos_pix[tuple(zip(*self.active_idxs))]/self.img_size
                self.init_grasp_state[:self.n_idxs, :2] = self.nn_bd_pts/self.img_size
            else:
                _, final_nn_bd_pts = self.nn_helper.get_min_dist(bd_cluster_centers, self.active_idxs, self.actions)
                self.final_state[:self.n_idxs, :2] = [self.convert_pix_2_world(bd_pts) for bd_pts in final_nn_bd_pts]
                if self.hp_dict['robot_frame']:
                    self.final_state[:self.n_idxs, :2] -= self.nn_helper.rb_pos_raw[tuple(zip(*self.active_idxs))]

                self.final_state[:self.n_idxs, 4:6] += self.actions[:self.n_idxs]

    # def set_final_state_variables_sim(self, goal_pos, init_pos, rot, final=False):
    #     delta_com = goal_pos - init_pos
    #     goal_nn_bd_pts_world_sim = [self.convert_real_2_sim(self.convert_pix_2_world(bd_pts)) for bd_pts in self.nn_bd_pts.copy()]
    #     goal_nn_bd_pts_world_sim = geom_utils.transform_pts_wrt_com(goal_nn_bd_pts_world_sim, (*delta_com, rot), init_pos)

    #     self.init_state[:self.n_idxs, 2:4] = goal_nn_bd_pts_world_sim
    #     self.final_state[:self.n_idxs, 2:4] = goal_nn_bd_pts_world_sim

    #     raw_rb_pos = self.nn_helper.rb_pos_world_sim[tuple(zip(*self.active_idxs))]
    #     if self.hp_dict['robot_frame']:
    #         self.init_state[:self.n_idxs, :2] -= raw_rb_pos
    #         self.init_state[:self.n_idxs, 2:4] -= raw_rb_pos
    #         self.final_state[:self.n_idxs, 2:4] -= raw_rb_pos
    #     else:
    #         self.init_state[:self.n_idxs, 4:6] = raw_rb_pos
    #         self.final_state[:self.n_idxs, 4:6] = raw_rb_pos

    #     return goal_nn_bd_pts_world_sim
    
    def test_grasping_policy(self, reset_after=False):
        self.get_nearest_robots_and_state(final=False)
        for i, idx in enumerate(self.active_idxs):
            self.actions_grasp[i] = self.pretrained_agent.get_actions(self.init_grasp_state[i], deterministic=True)

        self.init_state[:self.n_idxs, 4:6] += self.actions_grasp[:self.n_idxs]

        for i, idx in enumerate(self.active_idxs):
            # print(f'Robot {idx} is moving to {self.actions_grasp[i]}')
            traj = [[100*self.actions_grasp[i][0], -100*self.actions_grasp[i][1], low_z] for _ in range(20)]
            traj = self.practicalize_traj(traj)
            self.delta_agents[self.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx])
        for i in self.active_IDs:
            self.delta_agents[i-1].move_useful()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")
        if reset_after:
            self.reset()

    def set_goal_pose(self, sim=False):
        # cont = ''
        # while cont != 'c':
        #     cont = input("Press c to continue: ")
        self.goal_bd_pts, seg_map = self.get_seg_and_bd_pts()
        self.goal_bd_pts = np.array([self.convert_real_2_sim(self.convert_pix_2_world(bdpts)) for bdpts in self.goal_bd_pts])
        self.goal_pose = np.array(np.mean(self.goal_bd_pts, axis=0))

    def set_final_state_variables(self, goal_pos, init_pos, rot):
        delta_com = goal_pos - init_pos
        nn_bd_pts_world = [self.convert_real_2_sim(self.convert_pix_2_world(bd_pts)) for bd_pts in self.nn_bd_pts.copy()]
        nn_bd_pts_world = geom_utils.transform_pts_wrt_com(nn_bd_pts_world, (*delta_com, rot), init_pos)

        self.init_state[:self.n_idxs, 2:4] = nn_bd_pts_world
        self.final_state[:self.n_idxs, 2:4] = nn_bd_pts_world
        raw_rb_pos = self.nn_helper.rb_pos_world_sim[tuple(zip(*self.active_idxs))]
        
        self.init_state[:self.n_idxs, :2] -= raw_rb_pos
        self.init_state[:self.n_idxs, 2:4] -= raw_rb_pos
        self.final_state[:self.n_idxs, 2:4] -= raw_rb_pos

        # r_poses = self.nn_helper.rb_pos_world_sim[tuple(zip(*self.active_idxs))]
        # init_pts = self.init_state[:self.n_idxs, :2].copy()
        # # init_bd_pts = np.array([self.convert_real_2_sim(self.convert_pix_2_world(bdpts)) for bdpts in init_bd_pts])
        # goal_bd_pts = self.init_state[:self.n_idxs, 2:4].copy()
        # # g_bd_pt2 = np.array([self.convert_real_2_sim(self.convert_pix_2_world(bdpts)) for bdpts in self.goal_bd_pts])
        # act_grsp = self.actions_grasp[:self.n_idxs].copy()
        # acts = self.actions[:self.n_idxs].copy()
        
        # plt.figure(figsize=(10,17.78))
        # plt.scatter(r_poses[:, 0], r_poses[:, 1], c='#880000ff')

        # # plt.scatter(g_bd_pt2[:, 0], g_bd_pt2[:, 1], c='#ffa50066')
        # # plt.scatter(init_bd_pts[:, 0], init_bd_pts[:, 1], c = '#00ff0066')
        # plt.scatter(init_pts[:, 0], init_pts[:, 1], c = '#00ff00ff')
        # plt.scatter(goal_bd_pts[:, 0], goal_bd_pts[:, 1], c='red')

        # plt.quiver(r_poses[:, 0], r_poses[:, 1], act_grsp[:, 0], act_grsp[:, 1], scale=0.5, scale_units='xy')
        # plt.quiver(init_pts[:, 0], init_pts[:, 1], acts[:, 0], acts[:, 1], scale=1, scale_units='xy')
        # plt.gca().set_aspect('equal')
        # plt.show()
        return nn_bd_pts_world
    
    
    def visual_servoing(self, get_actions=False):
        if not get_actions:
            self.set_goal_pose()
            cont = ''
            while cont != 'c':
                cont = input("Press c to continue: ")
            self.test_grasping_policy()

        start_bd_pts, seg_map = self.get_seg_and_bd_pts()
        start_bd_pts = np.array([self.convert_real_2_sim(self.convert_pix_2_world(bdpts)) for bdpts in start_bd_pts])
        self.init_pose = np.mean(start_bd_pts, axis=0)

        _, init_bd_pts = self.nn_helper.get_min_dist_world(start_bd_pts, self.active_idxs, self.actions_grasp[:self.n_idxs])
        self.rot = geom_utils.get_transform(self.goal_bd_pts, start_bd_pts)[2]
        goal_bd_pts = self.set_final_state_variables(self.goal_pose, self.init_pose, self.rot)
        
        displacement_vectors = goal_bd_pts - init_bd_pts
        actions = self.actions_grasp[:self.n_idxs] + displacement_vectors
        if get_actions:
            return actions

        self.actions[:self.n_idxs] = np.clip(actions, -0.03, 0.03)

        for i, idx in enumerate(self.active_idxs):
            print(f'Robot {idx} is moving to {self.actions[i]}')
            traj = [[100*self.actions[i][0], -100*self.actions[i][1], low_z] for _ in range(20)]
            # traj = self.practicalize_traj2(traj)
            self.delta_agents[self.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx])

        for i in self.active_IDs:
            self.delta_agents[i-1].move_useful()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")
        # self.reset()

    
    def diffusion_step(self, agent, act_gt):
        bs = 10
        states = self.state_scaler.transform(np.tile(self.init_state[:self.n_idxs][None,...], (bs, 1, 1)))
        states = torch.tensor(states, dtype=torch.float32)
        obj_names = np.repeat(np.array(self.obj_name), bs)
        obj_name_enc = torch.tensor(self.obj_name_encoder.transform(obj_names)).to(torch.int64)
        pos = torch.tensor(np.tile(self.pos[:self.n_idxs][None,...], (bs, 1, 1)), dtype=torch.int64)
        noise = torch.randn((bs, self.n_idxs, 2))
        denoised_actions = self.action_scaler.inverse_transform(agent.actions_from_denoising_diffusion(noise, states, obj_name_enc, pos).detach().cpu().numpy())
        actions = np.mean(denoised_actions, axis=0)
        self.actions[:self.n_idxs] = np.clip(actions, -0.03, 0.03)

        # po = pos[idx]
        act_grasp = self.actions_grasp[:self.n_idxs]
        r_poses = self.nn_helper.rb_pos_world_sim[tuple(zip(*self.active_idxs))]
        r_poses2 = r_poses + act_grasp
        init_pts = self.init_state[:self.n_idxs,:2] + r_poses
        goal_bd_pts = self.init_state[:self.n_idxs,2:4] + r_poses
        act = self.actions[:self.n_idxs]

        plt.figure(figsize=(10,17.78))
        plt.scatter(self.nn_helper.kdtree_positions_world_sim[:, 0], self.nn_helper.kdtree_positions_world_sim[:, 1], c='#ddddddff')
        plt.scatter(init_pts[:, 0], init_pts[:, 1], c = 'blue', label='Initial Points')
        plt.scatter(goal_bd_pts[:, 0], goal_bd_pts[:, 1], c='red', label='Goal Points')

        plt.quiver(r_poses[:, 0], r_poses[:, 1], act_grasp[:, 0], act_grasp[:, 1], color='#0000ff88',label='Act Grasp')
        plt.quiver(r_poses2[:, 0], r_poses2[:, 1], act[:, 0], act[:, 1], color='#ff0000aa',label='Acts' )
        plt.quiver(r_poses2[:, 0], r_poses2[:, 1], act_gt[:, 0], act_gt[:, 1], color='#aaff55aa',label='Act VS')

        plt.gca().set_aspect('equal')
        plt.show()
        
    def diffusion_policy(self):
        """ Here we're doing everything in sim coords. Hence no 100*s """
        self.set_goal_pose(sim=True)
        cont = ''
        while cont != 'c':
            cont = input("Press c to continue: ")
        self.test_grasping_policy()

        # start_bd_pts, seg_map = self.get_seg_and_bd_pts()
        # start_bd_pts = np.array([self.convert_real_2_sim(self.convert_pix_2_world(bdpts)) for bdpts in start_bd_pts])
        # self.init_pose = np.mean(start_bd_pts, axis=0)

        # _, init_nn_bd_pts = self.nn_helper.get_min_dist_world_sim(start_bd_pts, self.active_idxs, self.actions_grasp[:self.n_idxs])
        # self.rot = geom_utils.get_transform(self.goal_bd_pts, start_bd_pts)[2]
        # goal_nn_bd_pts = self.set_final_state_variables_sim(self.goal_pose, self.init_pose, self.rot)
        # displacement_vectors = goal_nn_bd_pts - init_nn_bd_pts
        # actions = self.actions_grasp[:self.n_idxs] + displacement_vectors
        actions = self.visual_servoing(get_actions=True)
        
        self.diffusion_step(self.agent, actions)

        for i, idx in enumerate(self.active_idxs):
            print(f'Robot {idx} is moving to {self.actions[i]}')
            traj = [[100*self.actions[i][0], -100*self.actions[i][1], low_z] for _ in range(20)]
            self.delta_agents[self.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx])

        for i in self.active_IDs:
            self.delta_agents[i-1].move_useful()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")
        # self.reset()

    def inverse_dynamics(self):
        pass