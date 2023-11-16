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
import wandb

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

import utils.nn_helper as helper
from utils.geometric_utils import icp

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

device = torch.device("cuda:0")
BUFFER_SIZE = 20

low_z = 12
high_z = 5.5

class DeltaArrayReal:
    def __init__(self, img_embed_model, transform, agent, num_tips = [8,8]):
        """ Main Vars """
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        
        """ Data Collection Vars """
        self.lower_green_filter = np.array([30, 5, 5])
        self.upper_green_filter = np.array([90, 255, 255])
        self.plane_size = np.array([(1.8, 1.9),(23.125, -36.225)])
        self.nn_helper = helper.NNHelper(self.plane_size, real_or_sim="real")
        
        """ Fingertip Vars """
        self.finger_positions_cm = np.zeros((8,8,2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if i%2!=0:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301 + 2.165)
                else:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301)
        self.KMeans = KMeans(n_clusters=64, random_state=69, n_init='auto')

        """ Real World Util Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()
        self.neighborhood_fingers = []
        self.active_robots = []
        self.active_IDs = set()
        self.actions = {}

        """ Camera Vars """
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        """ Visual Servoing and RL Vars """
        self.bd_pts = []
        self.current_scene_frame = []
        self.batch_size = 128
        self.exploration_cutoff = 250
        
        self.model = img_embed_model
        self.transform = transform
        self.agent = agent
        self.init_state = []
        self.action = []
        self.log_pi = []

        self.ep_rewards = []
        self.ep_reward = 0
        self.optimal_reward = 30
        self.rot_matrix_90 = np.array([[0, -1],[1, 0]])
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()

    def setup_delta_agents(self, obj_pos = None):
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
        # self.reset()
        return
    
    def reset(self):
        for i in set(self.RC.robo_dict_inv.values()):
            self.delta_agents[i-1].reset()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Resetting Delta Robots...")
        self.wait_until_done()
        print("Done!")

    def get_nearest_robot_and_state(self,final=False):
        """ 
        A helper function to get the nearest robot to the block and crop the image around it 
        Get camera image -> Segment it -> Convert boundary to cartesian coordinate space in mm ->
        Get nearest robot to the boundary -> Crop the image around the robot -> Resize to 224x224 ->
        Randomize the colors to get rgb image.

        Returns nearest boundary distance and state of the MDP
        """
        ret, img = self.cam.read()
        if not ret:
            return
        
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
        # seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        # plt.imshow(seg_map)
        # plt.scatter(boundary_pts[:,1], boundary_pts[:,0], c='g')
        if len(boundary_pts) < 64:
            self.dont_skip_episode = False
            return None, None
        else:
            kmeans = self.KMeans.fit(boundary_pts)
            cluster_centers = kmeans.cluster_centers_
            
            # This array stores the equally space-separated boundary points of the object
            self.bd_pts = cluster_centers
            # self.bd_pts = boundary_pts
            idxs, neg_idxs = self.nn_helper.get_nn_robots(self.bd_pts)
            self.active_robots = list(idxs)
            
            for idx in self.active_robots:
                self.actions[idx] = np.array((0,0))

            min_dists, xys = self.nn_helper.get_min_dist(self.bd_pts, self.active_robots, self.actions)
            # print(xys)
            # for i in range(len(self.active_robots)):
            #     robopos = self.nn_helper.robot_positions[self.active_robots[i]]
            #     plt.scatter(robopos[1], robopos[0], c='orange')
            # plt.scatter(self.nn_helper.robot_positions[0,0,0], self.nn_helper.robot_positions[0,0,1], c="yellow")
            # plt.scatter(xys[:,1], xys[:,0], c='r')
            # plt.show()
            xys = [torch.FloatTensor(np.array([xys[i][0]/1080, xys[i][1]/1920, self.nn_helper.robot_positions[self.active_robots[i]][0]/1080, self.nn_helper.robot_positions[self.active_robots[i]][1]/1920])) for i in range(len(xys))]
            return min_dists, xys
    
    def wait_until_done(self, topandbottom=False):
        done_moving = False
        while not done_moving:
            # print(self.to_be_moved)
            for i in self.to_be_moved:
                try:
                    received = i.esp01.recv(BUFFER_SIZE)
                    ret = received.decode().strip()
                    # print(ret)
                    if ret == "A":
                        i.done_moving = True
                        time.sleep(0.5)
                except Exception as e: 
                    # print(e)
                    pass
            # print
            bool_dones = [i.done_moving for i in self.to_be_moved]
            # print(bool_dones)
            done_moving = all(bool_dones)
        time.sleep(0.5)
        for i in self.delta_agents:
            self.delta_agents[i].done_moving = False
        del self.to_be_moved[:]
        self.active_IDs.clear()
        # print("Done!")
        return
    
    def practicalize_traj(self, traj):
        traj[0] = [-1*traj[0][0], -1*traj[0][1], high_z]
        for i in range(1,5):
            traj[i] = [-1*traj[i][0], -1*traj[i][1], low_z]
        return traj
        
    
    def test_grasping_policy(self):
        # Set all robots to high pose

        # Capture an image and preprocess and extract neighbors and their closest points
        _, xys = self.get_nearest_robot_and_state()

        # Pass closest points through a policy
        for n, idx in enumerate(self.active_robots):
            self.actions[idx] = 100*self.agent.test_policy(xys[n])
            
            # Execute action given by the policy in real world
            print(f'Robot {idx} is moving to {self.actions[idx]}')
            traj = [[self.actions[idx][0], -1*self.actions[idx][1], low_z] for _ in range(20)]
            traj = self.practicalize_traj(traj)
            self.delta_agents[self.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            _ = [self.active_IDs.add(self.RC.robo_dict_inv[idx]) for i in self.active_robots]
            
        for i in self.active_IDs:
            self.delta_agents[i-1].move_useful()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")
        self.reset()