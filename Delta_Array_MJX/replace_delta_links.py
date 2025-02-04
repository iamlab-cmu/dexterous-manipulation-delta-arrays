import numpy as np
import time
import pickle as pkl
import time
import socket
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
np.set_printoptions(precision=4)
import threading
import math

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

LOW_Z = 9.8
MID_Z = 7.5
HIGH_Z = 5.5
BUFFER_SIZE = 20

class DeltaArrayReal:
    def __init__(self):
        self.max_agents = 64

        """ Real World Util Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()
        self.active_idxs = []
        self.active_IDs = set()
        self.n_idxs = 0
        self.all_robots = np.arange(64)
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()

    def setup_delta_agents(self):
        self.delta_agents = {}
        for i in range(1, 17):
            try:
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.05)
                self.delta_agents[i-1] = DeltaArrayAgent(esp01, i)
            except Exception as e:
                print("Error at robot ID: ", i)
                raise e
        # self.reset()
        
    def move_robots(self, active_idxs, actions, z_level, practicalize=False):
        for i, idx in enumerate(active_idxs):
            traj = [[actions[i][0], -actions[i][1], z_level] for _ in range(20)]
            # if practicalize:
            #     traj = self.practicalize_traj(traj)
            idx2 = (idx//8, idx%8)
            self.delta_agents[self.RC.robo_dict_inv[idx2] - 1].save_joint_positions(idx2, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx2])

        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots...")
        self.wait_until_done()
        print("Done!")

    def reset(self):
        for i in set(self.RC.robo_dict_inv.values()):
            self.delta_agents[i-1].reset()
            self.to_be_moved.append(self.delta_agents[i-1])

        print("Resetting Delta Robots...")
        self.wait_until_done()
        print("Done!")
    
    def wait_until_done(self, topandbottom=False):
        done_moving = False
        start_time = time.time()
        while not done_moving:
            for i in self.to_be_moved:
                try:
                    received = i.esp01.recv(BUFFER_SIZE)
                    ret = received.decode().strip()
                    if ret == "A":
                        i.done_moving = True
                except Exception as e:
                    time.sleep(0.1)
                    pass
                
            bool_dones = [i.done_moving for i in self.to_be_moved]
            done_moving = all(bool_dones)
            # Break if no communication happens in 15 seconds
            if time.time() - start_time > 15:
                print("Timeout exceeded while waiting for agents to complete.")
                done_moving = True
        time.sleep(0.1)
        for i in self.delta_agents:
            self.delta_agents[i].done_moving = False
        del self.to_be_moved[:]
        self.active_IDs.clear()
        

if __name__ == "__main__":
    delta_array = DeltaArrayReal()

    # Specify which robot to move (0-63)
    robot_idx = 27  # Change this to whatever robot you want to move

    # Create actions array with desired x,y coordinates (0,0)
    # Note that actions should be in the format [x, y]
    action = np.array([[0, 0]])  # This will be scaled by 100 inside move_robots()
 
    # Call move_robots with a single robot index and its action
    delta_array.move_robots([robot_idx], action, z_level=11.2)