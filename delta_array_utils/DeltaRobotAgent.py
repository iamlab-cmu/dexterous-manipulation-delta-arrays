import delta_array_utils.delta_trajectory_pb2 as delta_trajectory_pb2
import numpy as np
from serial import Serial
from math import *
import time
from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
import delta_array_utils.get_coords

NUM_MOTORS = 12
NUM_AGENTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
BUFFER_SIZE = 20
s_p = 1.5 #side length of the platform
s_b = 4.3 #side length of the base
l = 4.5 #length of leg attached to platform

Delta = Prismatic_Delta(s_p, s_b, l)
RC = RoboCoords()

class DeltaArrayAgent:
    def __init__(self, ser, robot_id):
        self.esp01 = ser
        self.delta_message = delta_trajectory_pb2.DeltaMessage()
        self.delta_message.id = robot_id
        self.delta_message.request_joint_pose = False
        self.delta_message.request_done_state = False
        self.delta_message.reset = False
        self.min_joint_pos = 0.005
        self.max_joint_pos = 0.0985

        self.done_moving = False
        self.current_joint_positions = [0.05]*12

    # Generate reset point for useless robots
    def reset(self):
        for j in range(20):
            ee_pts = [0,0,5.5]
            pts = Delta.IK(ee_pts)
            pts = np.array(pts) * 0.01
            pts = np.clip(pts,0.005,0.095)
            # jts = []
            # _ = [[jts.append(pts[j]) for j in range(3)] for i in range4]
            _ = [self.delta_message.trajectory.append(pts[i%3]) for i in range(12)]
        self.send_proto_cmd()
        del self.delta_message.trajectory[:]

    # Move useful robots
    def move_useful(self, pos):
        iks = [Delta.IK([0,0,5.5]) for i in range(20)]
        zeros = np.clip(np.array(iks) * 0.01,self.min_joint_pos,self.max_joint_pos)
        
        # Shape of pos -> (20, 3)
        iks = [Delta.IK(pos[i]) for i in range(20)]
        non_zeros = np.clip(np.array(iks) * 0.01,self.min_joint_pos,self.max_joint_pos)
        
        if self.delta_message.id == 11:
            iks2 = [Delta.IK([pos[i][0] + 0.63,pos[i][1] - 0.4,pos[i][2]]) for i in range(20)]
            non_zeros2 = np.clip(np.array(iks) * 0.01,self.min_joint_pos,self.max_joint_pos)
            # jts = np.hstack([non_zeros, zeros, zeros, non_zeros2])
            jts = np.hstack([zeros, non_zeros, non_zeros2, zeros])
        elif self.delta_message.id == 10:
            iks2 = [Delta.IK([pos[i][0] - 0.866,pos[i][1] + 0.5,pos[i][2]+0.2]) for i in range(20)]
            non_zeros2 = np.clip(np.array(iks) * 0.01,self.min_joint_pos,self.max_joint_pos)
            # jts = np.hstack([non_zeros, zeros, zeros, zeros])
            jts = np.hstack([zeros, non_zeros2, non_zeros, zeros])
        elif self.delta_message.id == 15:
            # jts = np.hstack([non_zeros, zeros, zeros, non_zeros])
            jts = np.hstack([zeros, non_zeros, non_zeros, zeros])
        elif self.delta_message.id == 14:
            iks2 = [Delta.IK([pos[i][0] + 1.125,pos[i][1] - 0.65,pos[i][2]+0.2]) for i in range(20)]
            non_zeros2 = np.clip(np.array(iks2) * 0.01,self.min_joint_pos,self.max_joint_pos)
            jts = np.hstack([zeros, non_zeros2, non_zeros, zeros])
            # jts = np.hstack([non_zeros, zeros, zeros, zeros])
        else: raise ValueError("Invalid robot ID")

        for j in range(20):
            _ = [self.delta_message.trajectory.append(jts[j][i]) for i in range(12)]
        self.send_proto_cmd()
        # print(jts, jts.shape, np.min(jts), np.max(jts))
        del self.delta_message.trajectory[:]
        # time.sleep(1)

    def stop(self):
        self.esp01.send()

    # def proto_clear(self):
    #     self.delta_message.Clear()
    #     self.delta_message.id = self.delta_message.id
    #     self.delta_message.request_joint_pose = self.delta_message.request_joint_pose
    #     self.delta_message.request_done_state = self.delta_message.request_done_state
    #     self.delta_message.reset = self.delta_message.reset


    def send_proto_cmd(self, ret_expected = False):
        serialized = self.delta_message.SerializeToString()
        self.esp01.send(b'\xa6~~'+ serialized + b'\xa7~~\r\n')
        if ret_expected:
            done_moving = self.esp01.recv(BUFFER_SIZE)
            print(done_moving, type(done_moving))
            return done_moving


    def move_joint_trajectory(self, desired_trajectory):
        desired_trajectory = np.clip(desired_trajectory,self.min_joint_pos,self.max_joint_pos)
        # Add joint positions to delta_message protobuf
        for i in range(20):
            _ = [self.delta_message.trajectory.append(desired_trajectory[i, j]) for j in range(12)]
        # self.send_proto_cmd()
        # print(self.delta_message)
        del self.delta_message.trajectory[:]

    def close(self):
        self.esp01.close()

    def get_done_state(self):
        _ = [[self.delta_message.trajectory.append(0) for j in 12] for i in range(20)]
        self.delta_message.request_done_state = True
        done_moving = self.send_proto_cmd(ret_expected = True)
        del self.delta_message.trajectory[:]
        self.delta_message.request_joint_pose = False
        self.delta_message.request_done_state = False
        self.delta_message.reset = False
        return done_moving

    def get_joint_positions(self):
        _ = [self.delta_message.joint_pos.append(0.5) for i in range(12)]
        self.delta_message.request_done_state = True
        self.current_joint_positions = self.send_proto_cmd(True)
        del self.delta_message.joint_pos[:]
        self.delta_message.request_joint_pose = False
        self.delta_message.request_done_state = False
        self.delta_message.reset = False
        # print(self.current_joint_positions)