import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt
import socket

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
import delta_array_utils.get_coords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm
# from cam_utils.pose_estimation import goal_tvec


# goal_tvec = np.array(([1.0], [1.9], [25.3]))  

# plt.ion()
BUFFER_SIZE = 20
13
low_z = 11.5
high_z = low_z - 2.5

class DeltaArrayEnv():
    def __init__(self, active_robots):
        self.rot_30 = np.pi/6
        self.low_z = low_z
        self.high_z = high_z
        
        """ Delta Robots Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()
        self.active_robots = active_robots

        self.active_IDs = set([self.RC.robo_dict_inv[i] for i in self.active_robots])
        print(self.active_IDs)
        mean_pos = np.zeros((2))
        for i in self.active_robots:
            mean_pos += self.RC.robot_positions[i]/10
        self.centroid = mean_pos/len(self.active_robots)

        """ Setup Delta Robot Agents """
        self.delta_agents = {}

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

        # Move all useful robots to the bottom
        for robot in self.active_robots:
            # Compute all vectors from active robots to obj_pos, move opposite till end
            if obj_pos is None:
                vec = self.RC.get_dist_vec(self.centroid, self.RC.robot_positions[robot]/10)    # Output of this function is a unit vector in direction of the centroid wrt robot_pos
            else:
                vec = self.RC.get_dist_vec(obj_pos, self.RC.robot_positions[robot]/10)    # Output of this function is a unit vector in direction of the point wrt robot_pos             
            vec = vec * -2    # Reverse the direction of the unit vector and amplify amplitude by a scalar
            traj = [[0,0, self.low_z]]
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)

        for i in self.active_IDs:
            self.delta_agents[i - 1].reset()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Initializing Delta Robots...")
        self.wait_until_done()
        print("Done!")
        return



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
            bool_dones = [i.done_moving for i in self.to_be_moved]
            # print(bool_dones)
            done_moving = all(bool_dones)
        time.sleep(0.1)
        for i in self.delta_agents:

            self.delta_agents[i].done_moving = False
        del self.to_be_moved[:]
        # print("Done!")
        return  

    """ Block Utils """
    def get_block_pose(self):
        """ Get the block pose """
        boolvar = True
        while boolvar == True:
            try:
                rot_error, pos_error, done_dict = pickle.load(open("./cam_utils/pose.pkl", "rb"))
                boolvar = False
            except: pass
        return pos_error, rot_error, done_dict

    """ Comm Utils """
    def check_movement_done(self):
        """ Check if all the agents have reached their goal """
        for i in self.useful_agents.values():
            while not i.get_done_state():
                time.sleep(0.5)
                continue
        return True
