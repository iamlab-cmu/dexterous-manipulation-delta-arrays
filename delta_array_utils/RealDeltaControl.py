import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt
import socket

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
import delta_array_utils.get_coords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm
from delta_array_utils.dynamic_motion_primitives import DMP
# from cam_utils.pose_estimation import goal_tvec

plt.ion()

# goal_tvec = np.array(([1.0], [1.9], [25.3]))  

# plt.ion()
BUFFER_SIZE = 20

class DeltaRobotEnv():
    def __init__(self, skill, n_bfs = 5):
        self.robot_positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.target_block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.active_robots = [(1,2),(1,3),(1,4), (3,2),(3,3),(3,4)]
        # self.robot_ids = set(self.RC.robo_dict_inv[i] for i in self.active_robots)
        self.robot_ids = set([10, 11, 14, 15])
        self.skill = skill
        self.rot_30 = np.pi/6
        # self.rot_30 = 0
        self.low_z = 12
        self.high_z = 5.5
        self.n_bfs = n_bfs
        self.start = [0, 0]
        self.goal = [0.015, 0.02]
        self.dmp_x = DMP(n_bfs)
        self.dmp_y = DMP(n_bfs)
        self.dmp_x.set_task_params(self.start[0], self.goal[0], 5, 0.01)
        self.dmp_y.set_task_params(self.start[1], self.goal[1], 5, 0.01)

        
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
        self.dmp_x.set_task_params(self.start[0], self.goal[0], 5, 0.01)
        self.dmp_y.set_task_params(self.start[1], self.goal[1], 5, 0.01)
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

    def generate_trajectory(self):
        if self.skill == "skill1":
            y1, y2 = (self.action[0] + 1)/2*0.01 + 0, (self.action[1] + 1)/2*0.005 - 0.005

            """ Generate linear trajectory using goal given by REPS for front and back grippers """ 
            self.skill_traj = np.linspace([0, 0], [y1, 0], 20)
            self.skill_hold_traj = np.linspace([0, 0], [y2, 0], 20)
            pickle.dump([y1, y2], open("./data/real_skill1_vars.pkl", "wb"))
        elif self.skill == "tilt":
            pos_x = self.dmp_x.fwd_simulate(500, 150*np.array(self.action[:self.n_bfs]))
            pos_y = self.dmp_y.fwd_simulate(500, 150*np.array(self.action[self.n_bfs:]))
            
            pos_x2 = pos_x[::len(pos_x)//20]
            pos_y2 = pos_y[::len(pos_y)//20]
            # plt.plot(pos_x2, pos_y2)
            skill_traj = np.vstack([pos_x2, pos_y2]).T
            
            skill_traj[:, 0] = np.clip(skill_traj[:, 0], -0.005, 0.025)
            skill_traj[:, 1] = np.clip(skill_traj[:, 1], -0.005, 0.05)

            plt.plot(100*skill_traj[:, 0], 100*skill_traj[:, 1],color='purple', alpha=0.1)
            plt.savefig("traj.png")
            # print(f"Start: {skill_traj[0]}, Goal: {skill_traj[-1]}, Weights: {self.action}")
            self.data_dict["Trajectory"].append(skill_traj)
            for i in range(20):
                # xy = self.rotate(np.array((0,0)) - np.array((skill_traj[i][0], self.rot_30)), 0)
                self.skill_traj[i] = [100*skill_traj[i, 0], 0, self.low_z - 100*skill_traj[i][1]]
            self.skill_hold_traj = np.linspace([-0.3, 0, self.low_z], [-0.3, 0, self.low_z], 20)

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
        elif self.skill == "tilt":
            if done_dict["is_done"]:
                error = 10
            else: error = pos_error * -1 + rot_error * -3
        elif self.skill == "lift":
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