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

low_z = 13.5
high_z = low_z - 2.5

class DeltaRobotEnv():
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
        if self.active_robots[0] != -1:
            self.active_IDs = set([self.RC.robo_dict_inv[i] for i in self.active_robots])
        else:
            self.active_IDs = set(self.RC.robo_dict_inv.values())
        print(self.active_IDs)
        # mean_pos = np.zeros((2))
        # for i in self.active_robots:
        #     mean_pos += self.RC.robot_positions[i]/10
        # self.centroid = mean_pos/len(self.active_robots)

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
            traj = [[vec[0], vec[1], self.low_z]]
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)

        for i in self.active_IDs:
            self.delta_agents[i - 1].reset()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Initializing Delta Robots...")
        self.wait_until_done()
        print("Done!")
        return

    def get_bent(self, center, box_pos, pos0, pos1, pos2, pos3, interp = "linear"):
        midpt0 = pos1*-0.4
        midpt1 = self.RC.get_dist_vec(pos0-center, pos1, norm=True, angle=np.pi)*2
        midpt2 = self.RC.get_dist_vec(pos3, pos2-center, norm=True, angle=np.pi)*2
        
        # midpt1 = pos1*-1
        # midpt2 = pos3*-0.8
        # midpt3 = pos3*-0.4
        midpt4 = pos2-center + pos3*-0.3 # pos3*-0.3
        if (midpt1 == midpt2).all(): midpt2 += 0.001

        points = np.vstack([midpt0, midpt1, midpt2, midpt4])

        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        alpha = np.linspace(0, 1, 10)
        interpolator =  interp1d(distance, points, kind=interp, axis=0)
        interpolated_points = interpolator(alpha)
        interpolated_points = np.clip(interpolated_points, -2, 2)
        # interpolated_points = np.vstack([interpolated_points])
        return interpolated_points

    def get_bent2(self, center, box_pos, pos0, pos1, pos2, pos3, interp = "linear"):
        # midpt1 = self.RC.get_dist_vec(box_pos, pos0-center, norm=True, angle=np.pi)*2
        # midpt2 = self.RC.get_dist_vec(pos2-center, box_pos, norm=True, angle=np.pi)*2
        midpt1 = pos0-center + pos1*-0.9
        midpt2 = pos0-center + pos1*-1.5
        midpt3 = pos2-center + pos3*-1.5
        midpt4 = pos2-center + pos3*-0.9
        if (midpt1 == midpt2).all(): midpt2 += 0.001
        elif (midpt2 == midpt3).all(): midpt3 += 0.001
        if (midpt3 == midpt4).all(): midpt4 += 0.001
        if (midpt4 == pos2-center).all(): pos2 += 0.001

        points = np.vstack([pos0-center, midpt1, midpt2, midpt3, midpt4, pos2-center])
        # points = np.vstack([midpt0, midpt1, midpt2, midpt4])

        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        alpha = np.linspace(0, 1, 10)
        interpolator =  interp1d(distance, points, kind=interp, axis=0)
        interpolated_points = interpolator(alpha)
        # interpolated_points = np.clip(interpolated_points, -2, 2)
        # print(interpolated_points.shape)  
        interpolated_points = np.clip(interpolated_points, -2, 2)
        # interpolated_points[:,0] = np.clip(interpolated_points[:,0], -2, 2)
        # interpolated_points[:,1] = np.clip(interpolated_points[:,1], -2, 0.7)
        # interpolated_points = np.vstack([interpolated_points, midpt4, pos2-center])
        return interpolated_points

    def set_init_finga(self, init_robots):
        traj = []
        traj.append([0, 0, self.low_z])
        for n, robot in enumerate(init_robots):
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
        
        for robot in init_robots:
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[self.RC.robo_dict_inv[robot] - 1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")

    def post_traj(self, traj, pos, n, robot, obj_pose, traj2, interp = "linear", zval=high_z):
        temp = []
        if (pos[0]!=-7777) and (traj2 is not None) and (traj2[n, 0] != -7777):
            if zval==self.high_z:
                curve = self.get_bent2(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], pos[:2], pos[3:5], traj2[n, :2],traj2[n, 3:5], interp)
            else:
                curve = self.get_bent2(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], pos[:2], pos[3:5], traj2[n, :2],traj2[n, 3:5], interp)
            temp.append([curve[0][0], curve[0][1], self.low_z])
            _ = [temp.append([vec[0], vec[1], zval]) for vec in curve[1:-1]]
            temp.append([curve[-1][0], curve[-1][1], self.low_z])
        elif ((traj2 is not None) and (traj2[n, 0] == -7777)):
            if zval!=self.high_z:
                vec = self.RC.get_dist_vec(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], norm=True)*1.5
            else:
                vec = pos[3:5]*-1.5
            temp.append([vec[0], vec[1], self.low_z])
            temp.append([vec[0], vec[1], zval])
            # temp.append([vec[0], vec[1], self.low_z])
        return temp

    def post_traj_new(self, traj, pos, n, robot, obj_pose, traj2, interp = "linear", zval=high_z):
        temp = []
        if (pos[0]!=-7777) and (traj2 is not None) and (traj2[n, 0] != -7777):
            if zval==self.high_z:
                curve = self.get_bent2(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], pos[:2], pos[3:5], traj2[n, :2],traj2[n, 3:5], interp)
            else:
                curve = self.get_bent2(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], pos[:2], pos[3:5], traj2[n, :2],traj2[n, 3:5], interp)
            temp.append([curve[0][0], curve[0][1], self.low_z])
            _ = [temp.append([vec[0], vec[1], zval]) for vec in curve[1:-1]]
            temp.append([curve[-1][0], curve[-1][1],  self.low_z])
    
        elif (pos[0]==-7777) and ((traj2 is not None) and (traj2[n, 0] != -7777)):
            if zval!=self.high_z:
                vec = self.RC.get_dist_vec(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], norm=True)*1.5
            else:
                vec = traj2[n, :2]-self.RC.robot_positions[robot]/10 + traj2[n, 3:5]*-0.5
                # vec = pos[3:5]*-1.5
            temp.append([vec[0], vec[1], zval])
            _ = [temp.append([vec[0], vec[1], zval]) for i in range(9)]
            temp.append([vec[0], vec[1],  self.low_z])
            # temp.append([vec[0], vec[1], self.low_z])
        
        elif (pos[0]!=-7777) and (traj2 is not None) and (traj2[n, 0] == -7777):
            vec = pos[:2]-self.RC.robot_positions[robot]/10 + pos[3:5]*-1
            temp.append([vec[0], vec[1], self.low_z])
            _ = [temp.append([vec[0], vec[1], self.high_z]) for i in range(10)]

        elif (pos[0]==-7777) and (traj2 is not None) and (traj2[n, 0] == -7777):
            _ = [temp.append([0, 0, self.high_z]) for i in range(11)]
        return temp

    def post_traj_inhand(self, traj, pos, n, robot, obj_pose, traj2, interp = "linear", zval=high_z):
        temp = []
        if (pos[0]!=-7777) and (traj2 is not None) and (traj2[n, 0] != -7777):
            vec = pos[:2] - self.RC.robot_positions[robot]/10
            _ = [temp.append([vec[0], vec[1], self.low_z]) for i in range(11)]
    
        elif (pos[0]==-7777) and ((traj2 is not None) and (traj2[n, 0] != -7777)):
            if zval!=self.high_z:
                vec = self.RC.get_dist_vec(self.RC.robot_positions[robot]/10, obj_pose[-1, :2], norm=True)*1.5
            else:
                vec = traj2[n, :2]-self.RC.robot_positions[robot]/10 + traj2[n, 3:5]*-0.5
                # vec = pos[3:5]*-1.5
            temp.append([vec[0], vec[1], zval])
            _ = [temp.append([vec[0], vec[1], zval]) for i in range(9)]
            temp.append([vec[0], vec[1],  self.low_z])
            # temp.append([vec[0], vec[1], self.low_z])
        
        elif (pos[0]!=-7777) and (traj2 is not None) and (traj2[n, 0] == -7777):
            vec = pos[:2] - self.RC.robot_positions[robot]/10
            _ = [temp.append([vec[0], vec[1], self.low_z]) for i in range(11)]

        elif (pos[0]==-7777) and (traj2 is not None) and (traj2[n, 0] == -7777):
            _ = [temp.append([0, 0, self.high_z]) for i in range(11)]
        return temp

    def set_plan(self, plan, obj_pose = None, traj2 = None):
        print(f"Active Robots {self.active_robots}")
        for n, robot in enumerate(self.active_robots):
            traj = []
            # if robot == (1,1):
            #     if plan[0,n][0]==-7777:
            #         traj.append([2, 1, self.high_z])
            # if plan[0,n][0]!=-7777:
            #     # vec = self.RC.get_dist_vec(self.RC.robot_positions[robot]/10, plan[0,n][3:5], norm=True, angle=0)*1.5
            #     vec = (plan[0,n][3:5]/np.linalg.norm(plan[0,n][3:5]))*-1.5
            #     traj.append([vec[0], vec[1], self.high_z])
            for pos in plan[:, n]:
                # print(f"POS: {pos[:2]}, RoboPos: {self.RC.robot_positions[robot]}")
                # is pos==-7777 it was NaN in the motion planner output.
                # print(pos)
                if pos[0] == -7777 and obj_pose is not None:
                    # print(obj_pose)
                    vec = self.RC.get_dist_vec(obj_pose[0, :2], self.RC.robot_positions[robot]/10, norm=True)
                    vec = vec * -1.5
                    traj.append([vec[0], vec[1], self.high_z])
                    break
                else:
                    vec = self.RC.get_dist_vec(pos[:2], self.RC.robot_positions[robot]/10, norm=False, angle=0)
                    print(f"Dist: {vec}")
                    traj.append([vec[0], vec[1], self.low_z])
            else:
                # This else.. continue prevents break from breaking all nested loops
                # self.post_traj(traj, pos, n, robot, obj_pose, traj2, interp="cubic", zval=self.low_z)
                self.post_traj(traj, pos, n, robot, obj_pose, traj2)
                print("Traj len: ", len(traj))
                self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
                # print("Traj: ", traj)
                continue
            
            self.post_traj(traj, pos, n, robot, obj_pose, traj2)
            print("Traj len: ", len(traj))
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
        
        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")

    def set_init_plan(self, obj_pose):
        for n, robot in enumerate(self.active_robots):
            traj = []
            if obj_pose is not None:
                # print(obj_pose)
                vec = self.RC.get_dist_vec(obj_pose[:2], self.RC.robot_positions[robot]/10, norm=True)
                vec = vec * -1
                traj.append([vec[0], vec[1], self.low_z])
                break
            
            # self.post_traj(traj, pos, n, robot, obj_pose, traj2, interp="linear", zval=self.low_z)
            # print("Traj len: ", len(traj))
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
        
        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")

    def set_plan_horizontal(self, plan, obj_pose = None, traj2 = None):
        print(f"Active Robots {self.active_robots}")
        for n, robot in enumerate(self.active_robots):
            traj = []
            
            # if plan[0,n][0]!=-7777:
            #     # vec = self.RC.get_dist_vec(self.RC.robot_positions[robot]/10, plan[0,n][3:5], norm=True, angle=0)*1.5
            #     vec = (plan[0,n][3:5]/np.linalg.norm(plan[0,n][3:5]))*-0.5
            #     traj.append([vec[0], vec[1], self.low_z])
            for pos in plan[:, n]:
                if pos[0] == -7777 and obj_pose is not None:
                    # print("HAKUNA MATATA")
                    vec = self.RC.get_dist_vec(obj_pose[0, :2], self.RC.robot_positions[robot]/10, norm=True)
                    vec = vec * -1.5
                    # traj.append([vec[0], vec[1], self.low_z])
                    traj.append([vec[0], vec[1], self.high_z])
                    # break
                else:
                    # vec = self.RC.get_dist_vec(pos[:2], self.RC.robot_positions[robot]/10, norm=False, angle=0)
                    vec = (pos[:2] - self.RC.robot_positions[robot]/10)/2 + 0.4*pos[3:5]
                    print(f"Dist: {vec}")
                    traj.append([vec[0], vec[1], self.low_z])
            
            traj.extend(self.post_traj_new(traj, pos, n, robot, obj_pose, traj2, interp="linear", zval=self.high_z))
            print("Traj len: ", len(traj))
            # print("Traj: ", traj)
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
        
        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots on Trajectory...")
        self.wait_until_done()
        print("Done!")

    
    def set_plan_inhand(self, plan, obj_pose = None, traj2 = None):
        print(f"Active Robots {self.active_robots}")
        for n, robot in enumerate(self.active_robots):
            traj = []
            for pos in plan[:, n]:
                if pos[0] == -7777 and obj_pose is not None:
                    vec = self.RC.get_dist_vec(obj_pose[0, :2], self.RC.robot_positions[robot]/10, norm=True)
                    vec = vec * -1.5
                    # traj.append([vec[0], vec[1], self.low_z])
                    traj.append([vec[0], vec[1], self.high_z])
                else:
                    # vec = self.RC.get_dist_vec(pos[:2], self.RC.robot_positions[robot]/10, norm=False, angle=0)
                    vec = (pos[:2] - self.RC.robot_positions[robot]/10)/2 #+ 0.4*pos[3:5]
                    print(f"Dist: {vec}")
                    traj.append([vec[0], vec[1], self.low_z])
            
            traj.extend(self.post_traj_new(traj, pos, n, robot, obj_pose, traj2, interp="linear", zval=self.high_z))
            print("Traj len: ", len(traj))
            # print("Traj: ", traj)
            self.delta_agents[self.RC.robo_dict_inv[robot] - 1].save_joint_positions(robot, traj)
        
        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots on Trajectory...")
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
                        time.sleep(0.5)
                except Exception as e: 
                    # print(e)
                    pass
            bool_dones = [i.done_moving for i in self.to_be_moved]
            # print(bool_dones)
            done_moving = all(bool_dones)
        time.sleep(0.5)
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

if __name__=="__main__":
    env = DeltaRobotEnv(active_robots=[-1])
    print("Done")