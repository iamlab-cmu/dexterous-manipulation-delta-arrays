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
# from cam_utils.pose_estimation import goal_tvec


# goal_tvec = np.array(([1.0], [1.9], [25.3]))  

# plt.ion()
BUFFER_SIZE = 20

class DeltaRobotEnv():
    def __init__(self, skill):
        self.robot_positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.target_block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.active_robots = [(1,2),(1,3),(1,4), (3,2),(3,3),(3,4)]
        # self.robot_ids = set(self.RC.robo_dict_inv[i] for i in self.active_robots)
        self.robot_ids = set([10, 11, 14, 15])
        self.skill = skill
        self.rot_30 = np.pi/6
        # self.rot_30 = 0
        self.low_z = 12.2
        self.high_z = 5.5
        
        """ Delta Robots Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.RC = RoboCoords()

        """ RL Vars """
        self.return_vars = {"observation": None, "reward": None, "done": None, "info": {"is_solved": False}}
        self.trajectories = []
        self.prev_action = None
        self.skill_thresh = np.array((0.2, 1.2))
        self.skill_traj = list(np.zeros((20,3)))
        self.skill_hold_traj = list(np.zeros((20,3)))
        self.data_dict = {"Points": [], "Trajectory": [], "Reward": []}
        self.left_top_pos = np.zeros((20,3))
        self.left_top_pos[0:2, :] = [-1.3,0,self.low_z-2]
        self.left_top_pos[2:18, :] = [*self.rotate(np.array((0.6, 0)), 0), self.low_z]
        self.left_top_pos[18:, :] = [-1.3,0,self.low_z]

        if self.skill != "lift":
            self.right_top_pos = [[0,0,self.low_z] for i in range(20)]
        else:
            self.right_top_pos = np.zeros((20,3))
            self.right_top_pos[0:2, :] = [1.3,0,self.low_z-2]
            self.right_top_pos[2:18, :] = [*self.rotate(np.array((-0.6, 0)), 0), self.low_z]
            self.right_top_pos[18:, :] = [1.3,0,self.low_z]

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
        for i in range(1, 17):
            if i!=9:
                # Get IP Addr and socket of each grid and classify them as useful or useless
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.1)
                if i not in self.robot_ids:
                    self.useless_agents.append(DeltaArrayAgent(esp01, i))
                else:
                    self.useful_agents[i] = DeltaArrayAgent(esp01, i)
        
        # Move all useless robots to the top 
        for i in self.useless_agents:
            i.reset()

        # Move all useful robots to the bottom
        for i in self.useful_agents.values():
            pos = [[0,0,self.low_z] for i in range(20)]
            i.move_useful(pos)
            self.to_be_moved.append(i)
        print("Initializing Delta Robots...")
        # self.check_movement_done()
        self.wait_until_done()
        return

    def step(self, action):
        self.action = action
        self.generate_trajectory()
        # print("Trajectory Length: ", len(self.skill_traj))
        self.move_top_and_bottom_agents()
        # print("Stepping into Environment...")
        self.wait_until_done()
        # """ Get Rewards and Plug Into REPS """
        self.get_reward()
        return self.return_vars["observation"], self.return_vars["reward"], self.return_vars["done"], self.return_vars["info"]

    def move_top_and_bottom_agents(self):
        for i in self.useful_agents.values():
            if i.delta_message.id not in [14, 15]:
                i.move_useful(self.skill_hold_traj)
            else:
                i.move_useful(self.skill_traj)
                self.to_be_moved.append(i)
        return

    def reset(self):
        """ Push the block towards the back a little and retract the fingers """
        print("Resetting Delta Robots...")
        for i in self.useful_agents.values():
            self.to_be_moved.append(i)
            if i.delta_message.id not in [14, 15]:
                i.move_useful(self.right_top_pos)
            else:
                i.move_useful(self.left_top_pos)
        self.wait_until_done()
    # def reset(self):
    #     """ Push the block towards the back a little and retract the fingers """
    #     print("Reset HAKUNA MATATA")
    #     top_pos = np.zeros((20,3))
    #     top_pos[:,:] = [0,0,12]
    #     for i in self.useful_agents.values():
    #         if i.delta_message.id not in [11, 10]:
    #             pos = [[0,0,12] for i in range(20)]
    #             i.move_useful(pos)
    #         else:
    #             i.move_useful(top_pos)
    #     top_pos[:,:] = [*self.rotate(np.array((0,2)), self.rot_30), 10]
    #     for i in self.useful_agents.values():
    #         if i.delta_message.id in [11, 10]:
    #             i.move_useful(top_pos)
    #     top_pos[:,:] = [0,0,12]
    #     for i in self.useful_agents.values():
    #         if i.delta_message.id in [11, 10]:
    #             i.move_useful(top_pos)
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
                        time.sleep(0.5)
                except Exception as e: 
                    # print(e)
                    pass
            done_moving = all([i.done_moving for i in self.to_be_moved])
        time.sleep(0.5)
        for i in self.useful_agents.values():
            i.done_moving = False
        del self.to_be_moved[:]
        # print("Done!")
        return 


    def Bezier_Curve(self, p1, p2, p3, p4):
        points = np.array((p1, p2, p3, p4))
        curve = np.array([(1-t)**3*p1 + 3*t*(1-t)**2*p2 + 3*t**2*(1-t)*p3 + t**3*p4 for t in np.linspace(0, 1, 20)]).T
        skill_traj = self.DMP_trajectory(curve)
        assert len(skill_traj) == 20

        self.data_dict['Points'].append(points)
        self.data_dict['Trajectory'].append(curve)
        plt.plot(curve[0], curve[1])
        plt.scatter(points[:, 0], points[:, 1])
        plt.savefig(f"./traj_imgs/{len(os.listdir('./traj_imgs'))}.png")
        return skill_traj

    def DMP_trajectory(self, curve):
        # Fill this function!!!!!!!!!!!!!!!!!
        DMP = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=20, ay=np.ones(2) * 10.0)
        DMP.imitate_path(y_des=curve)
        trajectory, _, _ = DMP.rollout(tau=1)
        trajectory = trajectory[::5]
        return trajectory

    def generate_trajectory(self):
        if self.skill == "skill1":
            y1, y2 = (self.action[0] + 1)/2*0.01 + 0, (self.action[1] + 1)/2*0.005 - 0.005

            """ Generate linear trajectory using goal given by REPS for front and back grippers """ 
            self.skill_traj = np.linspace([0, 0], [y1, 0], 20)
            self.skill_hold_traj = np.linspace([0, 0], [y2, 0], 20)
            pickle.dump([y1, y2], open("./data/real_skill1_vars.pkl", "wb"))
        elif self.skill == "tilt":
            x1, y1 = (self.action[0] + 1)/2*-1.5 + 0, (self.action[1] + 1)/2*1 + 0
            x2, y2 = (self.action[2] + 1)/2*-2.3 + 0, (self.action[3] + 1)/2*1 + 0
            x3, y3 = (self.action[4] + 1)/2*-2.2 - 0.25, (self.action[5] + 1)/2*4 + 0.5
            
            """ Uncomment or comment based on whether learning skill1 is required """
            # prev_x1, prev_y2 = pickle.load(open("./data/real_skill1_vars.pkl", "rb"))
            x0, y0 = 0, 0

            """ Generate Bezier curve trajectory using REPS variables and smoothen trajectory using DMP """
            skill_traj = self.Bezier_Curve(np.array([x0, y0]), np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]))
            skill_traj[:, 1] = np.clip(skill_traj[:, 1], -0.1, 6)
            skill_traj[:, 0] = np.clip(skill_traj[:, 0], -2.5, 2.5)
            for i in range(20):
                xy = self.rotate(np.array((0,0)) - np.array((skill_traj[i][0], self.rot_30)), 0)
                print([*xy, self.low_z - skill_traj[i][1]])
                self.skill_traj[i] = [*xy, self.low_z - skill_traj[i][1]]
            self.skill_hold_traj = np.linspace([0, 0], [y0, 0], 20)

        elif self.skill == "lift":
            # Left grippers
            x1, y1 = (self.action[0] + 1)/2*-0.6 + 0, (self.action[1] + 1)/2*0.0 + 0
            x2, y2 = (self.action[2] + 1)/2*-1.2 + 0, (self.action[3] + 1)/2*0.0 + 0
            x3, y3 = (self.action[4] + 1)/2*-1.3 - 0.25, (self.action[5] + 1)/2*2 + 0.5
            x0, y0 = 0.0, 0.0
            left_skill_traj = self.Bezier_Curve(np.array([x0, y0]), np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]))
            left_skill_traj[:, 1] = np.clip(left_skill_traj[:, 1], -0.1, 6)
            left_skill_traj[:, 0] = np.clip(left_skill_traj[:, 0], -2.5, 2.5)
            # Right grippers
            x1, y1 = (self.action[0] + 1)/2*0.6 + 0, (self.action[1] + 1)/2*0.5 + 0
            x2, y2 = (self.action[2] + 1)/2*1.2 + 0, (self.action[3] + 1)/2*0.5 + 0
            x3, y3 = (self.action[4] + 1)/2*1.3 + 0.25, (self.action[5] + 1)/2*3 + 0.5
            x0, y0 = 0.0, 0.0
            right_skill_traj = self.Bezier_Curve(np.array([x0, y0]), np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]))
            right_skill_traj[:, 1] = np.clip(right_skill_traj[:, 1], -0.2, 6)
            right_skill_traj[:, 0] = np.clip(right_skill_traj[:, 0], -2.5, 2.5)

            for i in range(20):
                xy = self.rotate(np.array((0,0)) - np.array((left_skill_traj[i][0], self.rot_30)), 0)
                self.skill_traj[i] = [*xy, self.low_z - left_skill_traj[i][1]]

                xy = self.rotate(np.array((0,0)) - np.array((right_skill_traj[i][0], self.rot_30)), 0)
                self.skill_hold_traj[i] = [*xy, self.low_z - right_skill_traj[i][1]]

        else:
            raise ValueError("Invalid skill Skill can be either skill1 or skill2")

    def get_reward(self):
        pos_error, rot_error, done_dict = self.get_block_pose()
        if self.skill == "skill1":
            if pos_error < self.thresh[0]:
                done_dict["is_done"] = True
                error = 5
            else: error = pos_error
        elif self.skill == "skill2":
            if done_dict["is_done"]:
                error = 10
            else: error = pos_error * -1 + rot_error * -3
        elif self.skill == "skill3":
            if done_dict["is_done"]:
                error = 10
            else: error = pos_error * -2 + rot_error * -2

        print(rot_error, pos_error, done_dict["is_done"])
        self.data_dict["Reward"].append(error)
        self.return_vars['reward'], self.return_vars['done'] = error, done_dict["is_done"]
        self.return_vars['info']["is_solved"] = self.return_vars['done']
        
        pickle.dump(self.data_dict, open("./data/traj_data.pkl", "wb"))
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

if __name__=="__main__":
    env = DeltaRobotEnv("skill1")
    print("Done")