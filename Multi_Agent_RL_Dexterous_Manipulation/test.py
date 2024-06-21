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
        
        """ Fingertip Vars """
        # Real World Y-Axis is -Y as opposed to +Y in simulation
        self.finger_positions_cm = np.zeros((8,8,2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if i%2!=0:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301 + 2.165)
                else:
                    self.finger_positions_cm[i][j] = (i*3.75, -j*4.3301)

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
    
if __name__=="__main__":
    start_capture_thread()
    ids = [(4, 4), (2, 4), (5, 4), (2, 3), (2, 2), (5, 3), (3, 2), (4, 1), (3, 5)]
    # ids = [(0, 0), (0, 1)]

    delta_array = DeltaArrayReal(agents=ids, objs=None, hp_dict=None)
    x, y = 1, 0
    while True:
        for i, idx in enumerate(ids):
            x *= -1
            y *= -1
            traj = [[x, y, low_z] for _ in range(20)]
            delta_array.delta_agents[delta_array.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            delta_array.active_IDs.add(delta_array.RC.robo_dict_inv[idx])

        for i in delta_array.active_IDs:
            delta_array.delta_agents[i-1].move_useful()
            delta_array.to_be_moved.append(delta_array.delta_agents[i-1])

        print("Moving Delta Robots on Trajectory...")
        delta_array.wait_until_done()
