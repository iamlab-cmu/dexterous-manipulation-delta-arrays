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
        self.skill = skill
        self.rot_30 = np.pi/6
        
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
        self.skill_thresh = np.array((0.1, 0.4))

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
        
    def rotate(self, vector, angle, plot=False):
        # Rotation from Delta Array axis to cartesian axis = 30 degrees. 
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        vector = vector@rot_matrix
        return vector

    def setup_delta_agents(self):
        self.delta_agents = []
        # Obtain numbers of 2x2 grids
        robot_ids = set(self.RC.robo_dict_inv[i] for i in self.active_robots)
        for i in range(1, 17):
            if i!=9:
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

        # Move all useful robots to the bottom
        for i in self.useful_agents.values():
            pos = [[0,0,9] for i in range(20)]
            i.move_useful(pos)
        time.sleep(2)
        return

    def step(self, action):
        self.action = action
        """ Make Robots Move Acc to Gen Trajectory """
        self.generate_trajectory()
        """ Write a more proper reset function """
        """ Get Rewards and Plug Into REPS """
        return self.return_vars["observation"], self.return_vars["reward"], self.return_vars["done"], self.return_vars["info"]

    def reset(self):
        """ Push the block towards the back a little and retract the fingers """
        top_pos = np.zeros((20,3))
        top_pos[:,:] = [0,0,12]
        for i in self.useful_agents.values():
            if i.delta_message.id not in [11, 10]:
                pos = [[0,0,12] for i in range(20)]
                i.move_useful(pos)
            else:
                i.move_useful(top_pos)
        top_pos[:,:] = [*self.rotate(np.array((0,2)), self.rot_30), 10]
        for i in self.useful_agents.values():
            if i.delta_message.id in [11, 10]:
                i.move_useful(top_pos)
        top_pos[:,:] = [0,0,12]
        for i in self.useful_agents.values():
            if i.delta_message.id in [11, 10]:
                i.move_useful(top_pos)

    def Bezier_Curve(self, t, p1, p2, p3, p4):
        return (1-t)**3*p1 + 3*t*(1-t)**2*p2 + 3*t**2*(1-t)*p3 + t**3*p4

    def DMP_trajectory(self, curve):
        # Fill this function!!!!!!!!!!!!!!!!!
        DMP = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=200, ay=np.ones(2) * 10.0)
        y_track = []
        dy_track = []
        ddy_track = []

        DMP.imitate_path(y_des=curve)
        trajectory, _, _ = DMP.rollout(tau=1)
        return trajectory

    def generate_trajectory(self):
        if self.skill == "skill1":
            y1, y2 = (self.action[0] + 1)/2*0.01 + 0, (self.action[1] + 1)/2*0.005 - 0.005

            """ Generate linear trajectory using goal given by REPS for front and back grippers """ 
            self.skill_traj = np.linspace([0, 0], [y1, 0], self.time_horizon - 1)
            self.skill_hold_traj = np.linspace([0, 0], [y2, 0], self.time_horizon - 1)
            pickle.dump([y1, y2], open("./data/real_skill1_vars.pkl", "wb"))
        elif self.skill == "skill2":
            x1, y1 = (self.action[0] + 1)/2*0.02 + 0, (self.action[1] + 1)/2*0.05 + 0
            x2, y2 = (self.action[2] + 1)/2*0.02 + 0, (self.action[3] + 1)/2*0.05 + 0
            x3, y3 = (self.action[4] + 1)/2*0.015 + 0.01, (self.action[5] + 1)/2*0.06 + 0.01
            prev_y1, prev_y2 = pickle.load(open("./data/real_skill1_vars.pkl", "rb"))

            """ Generate Bezier curve trajectory using REPS variables and smoothen trajectory using DMP """
            points = np.array(((prev_y1, 0), (x1, y1), (x2, y2), (x3, y3)))
            # print(points)
            curve = np.array([self.Bezier_Curve(t, *points) for t in np.linspace(0, 1, self.time_horizon - 1)]).T
            self.skill_traj = self.DMP_trajectory(curve)
            self.skill_hold_traj = np.linspace([0, 0], [prev_y2, 0], self.time_horizon - 1)

            # print(np.min(self.skill_traj[:,0]),np.max(self.skill_traj[:,0]),np.min(self.skill_traj[:,1]),np.max(self.skill_traj[:,1]))
            plt.plot(curve[0], curve[1])
            # plt.plot(self.skill_traj[:, 0], self.skill_traj[:, 1])
            plt.scatter(points.T[0], points.T[1])
            plt.savefig(f"./traj_imgs/{len(os.listdir('./real_traj_imgs'))}.png")
            # plt.show()
        else:
            raise ValueError("Invalid skill Skill can be either skill1 or skill2")

    def get_reward(self):
        pos_error, rot_error, done_dict = self.get_block_pose()
        if self.skill == "skill1":
            if pos_error < self.thresh[0]:
                done_dict["is_done"] = True
                error = 100
            else: error = pos_error * -10
        elif self.skill == "skill2":
            if done_dict["is_done"]:
                error = 100
            else: error = pos_error * -10 + rot_error * -40
        return error, done_dict

    """ Block Utils """
    def get_block_pose(self):
        """ Get the block pose """
        rot_error, pos_error, done_dict = pickle.load(open("./cam_utils/pose.pkl", "rb"))
        return pos_error, rot_error, done_dict

if __name__=="__main__":
    env = DeltaRobotEnv("skill1")
    print("Done")