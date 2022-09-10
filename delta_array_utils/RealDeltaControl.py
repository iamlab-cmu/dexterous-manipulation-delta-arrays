import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt
import socket

import pydmps
import pydmps.dmp_discrete

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
import delta_array_utils.get_coords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

plt.ion()

class DeltaRobotEnv():
    def __init__(self, skill):
        self.robot_positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.target_block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.active_robots = [(2,2),(2,3),(2,4), (4,2),(4,3),(4,4)]
        
        """ Delta Robots Vars """
        self.NUM_MOTORS = 12
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()

        """ RL Vars """
        self.return_vars = {"observation": None, "reward": None, "done": None, "info": {"is_solved": False}}
        self.trajectories = []
        self.prev_action = None

        """ Setup Delta Robot Agents """
        self.useless_agents = []
        self.useful_agents = {}
        self.setup_delta_agents()

    def context(self):
        # Some random function of no significance. But don't remove it!! Needed for REPS
        return None

    def store_trajectory(self):
        # Another random function of no significance. But don't remove it!! Needed for REPS
        self.trajectories.append(self.prev_action)

    def setup_delta_agents(self):
        self.delta_agents = []
        # Obtain numbers of 2x2 grids
        robot_ids = set(self.RC.robo_dict_inv[i] for i in self.active_robots)
        for i in range(1, 17):
            # Get IP Addr and socket of each grid and classify them as useful or useless
            ip_addr = srm.inv_delta_comm_dict[i]
            esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            esp01.connect((ip_addr, 80))
            if i not in robot_ids:
                self.useless_agents.append(DeltaArrayAgent(esp01, i))
            else:
                self.useful_agents[i] = DeltaArrayAgent(esp01, i)
        
        # Move all useless robots to the top 
        for i in self.useless_agents:
            i.reset()

        return

    def move_top_layer(self, position):
        return

if __name__=="__main__":
    env = DeltaRobotEnv("skill1")
    print("Done")