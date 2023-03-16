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
s_p = 1.5 #side length of the platform
s_b = 4.3 #side length of the base
l = 4.5 #length of leg attached to platform

Delta = Prismatic_Delta(s_p, s_b, l)
RC = RoboCoords()

# arduino = Serial('/dev/ttyACM0', 57600)
# arduino = Serial('COM4', 57600)
# delta_message = linear_actuator_pb2.lin_actuator()
# delta_message.id = 1

class DeltaArrayAgent:
    def __init__(self, ser, robot_id):
        self.arduino = ser
        self.delta_message = delta_trajectory_pb2.DeltaMessage()
        # self.joint_pos = delta_trajectory_pb2.JointPos()
        self.delta_message.id = robot_id
        self.delta_message.request_joint_pose = False
        self.delta_message.request_done_state = False
        self.delta_message.reset = False
        self.min_joint_pos = 0.005
        self.max_joint_pos = 0.0985

        self.done_moving = False

    # GENERATE RESET and STOP commands in protobuf
    def reset(self):
        self.arduino.write()

    def stop(self):
        self.arduino.write()

    # def proto_clear(self):
    #     self.delta_message.Clear()
    #     self.delta_message.id = self.delta_message.id
    #     self.delta_message.request_joint_pose = self.delta_message.request_joint_pose
    #     self.delta_message.request_done_state = self.delta_message.request_done_state
    #     self.delta_message.reset = self.delta_message.reset


    def send_proto_cmd(self, ret_expected = False):
        serialized = self.delta_message.SerializeToString()
        self.arduino.write(bytes(b'\xa6') + serialized + bytes(b'\xa7'))
        if ret_expected:
            """ DEAL WITH RETURN STUFF LATER """
            pass
            # reachedPos = str(self.arduino.readline())
            # # print(reachedPos.split(" "))
            # reachedPos = reachedPos.strip().split(" ")
            # if self.delta_message.id == int(reachedPos[0].split(':')[-1]):
            #     return [float(x) for x in reachedPos[1:-1]]
            # else:
            #     print("ERROR, incorrect robot ID requested.")
            #     return [0.05]*12


    def move_joint_position(self, desired_joint_positions):
        """ desired_joint_positions is a 50x12 numpy array"""
        desired_joint_positions = np.clip(desired_joint_positions,self.min_joint_pos,self.max_joint_pos)
        # Add joint positions to delta_message protobuf

        for i in range(50):
            self.joint_pos.joint_pos.extend(desired_joint_positions[i])
            self.delta_message.trajectory.append(self.joint_pos)

        self.send_proto_cmd()
        del self.delta_message.trajectory[:]

    def close(self):
        self.arduino.close()

    def get_joint_positions(self):
        self.delta_message.request_done_state = True
        self.current_joint_positions = self.send_proto_cmd(True)
        del self.delta_message.trajectory[:]
        self.delta_message.request_joint_pose = False
        self.delta_message.request_done_state = False
        self.delta_message.reset = False
        # print(self.current_joint_positions)


class DeltaArrayEnv:
    def __init__(self, port):
        self.ser = Serial(port, 57600)
        self.agents = {}
        for i in NUM_AGENTS:
            self.agents[i] = DeltaArrayAgent(self.ser, i)
        self.done_states = np.array([0]*12)
        # self.generate_gaits()
        self.lowz = 8
        self.highz = 11

    def reset(self):
        jts = []
        for i in range(0, 12):
            pt = Delta.IK((0,0,8))
            pt = np.array(pt)*0.01
            jts.extend(pt)
        
        for i in NUM_AGENTS:
            self.agents[i].move_joint_position(jts)


    def move_over_trajectory(self, traj = "verticle"):
        if traj == "circle":
            thetas = np.linspace(0, 2*np.pi, 10)
            r = 1
            for theta in thetas:
                # for z in zvals:
                ee_pts = [r*np.cos(theta), r*np.sin(theta), 10.0]
                pts = Delta.IK(ee_pts)
                pts = np.array(pts) * 0.01
                # print(theta)
                jts = []
                for i in range(0, 4):
                    for j in range(3):
                        jts.append(pts[j])
                
                for i in NUM_AGENTS:
                    self.agents[i].move_joint_position(jts)

        elif traj=="vertical":
            # pts = Delta.IK([0,0,10])
            pts1 = [0.05,0.05,0.05]
            pts2 = [0.02,0.02,0.02]
            jts = []
            for i in range(0, 4):
                if i==3:
                    jts.extend(pts1)
                else:
                    jts.extend(pts2)
                
            # print(jts)
            for i in NUM_AGENTS:
                self.agents[i].move_joint_position(jts)
                
            for i in NUM_AGENTS:
                self.agents[i].get_joint_positions()

    
        
    # def planar_translation(self, motion = "left", wall = []):
    #     a = 0
    #     up_movement, down_movement = RC.move_delta_array_planar(motion)
    #     lr_pts0 = [*up_movement, *down_movement, *down_movement, *up_movement]
    #     lr_pts1 = [*down_movement, *up_movement, *up_movement, *down_movement]
    #     ud_pts0 = [*up_movement, *up_movement, *down_movement, *down_movement]
    #     ud_pts1 = [*down_movement, *down_movement, *up_movement, *up_movement]
    #     if motion == "left" or motion == "right":
    #         while True:
    #             if a==0:
    #                 for i in NUM_AGENTS:
    #                     self.agents[i].move_joint_position(lr_pts0)
    #                 a=1
    #             else:
    #                 for i in NUM_AGENTS:
    #                     self.agents[i].move_joint_position(lr_pts1)
    #                 a=0
    #             time.sleep(1.5)

    #     if motion == "up" or motion == "down":
    #         while True:
    #             if a==0:
    #                 for i in NUM_AGENTS:
    #                     self.agents[i].move_joint_position(ud_pts0)
    #                 a=1
    #             else:
    #                 for i in NUM_AGENTS:
    #                     self.agents[i].move_joint_position(ud_pts1)
    #                 a=0
    #             time.sleep(1.5)

    def move_delta_array(self, point = (162.378,131.25), pattern="converge", angle=pi/4, wall=[], zmax=None, zmin=None):
        if pattern=="up":
            vecs = RC.get_dist_vec((162.378, 9999999999999999))
        elif pattern=="down":
            vecs = RC.get_dist_vec((162.378, 9999999999999999))
            vecs = RC.rotate(vecs, np.pi)
            vecs = RC.normalize_vec(vecs)
        elif pattern=="left":
            vecs = RC.get_dist_vec((162.378, 9999999999999999))
            vecs = RC.rotate(vecs, -np.pi/2)
            vecs = RC.normalize_vec(vecs)
        elif pattern=="right":
            vecs = RC.get_dist_vec((9999999999999999,131.25))
        elif pattern=="converge":
            vecs = RC.get_dist_vec(point)
        elif pattern=="rotate":
            vecs = RC.get_rot_vec(point, angle)

        RC.set_pattern(vecs, zmax, zmin)
        if len(wall)>0:
            RC.set_wall(wall)
        a = 0
        while True:
            if a==0:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=1
            else:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=0
            time.sleep(1.5)

    def grip_and_move_objects(self):
        RC.grip_pattern()
        a = 0
        while True:
            if a==0:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=1
            else:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=0
            time.sleep(0.5)

    def glamorous(self, wall, w_ht):
        RC.set_wall(wall, w_ht)
        a = 0
        while True:
            if a==0:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=1
            else:
                for i in NUM_AGENTS:
                    pts = RC.get_pattern(i,a)
                    self.agents[i].move_joint_position(pts)
                a=0
            time.sleep(4)


if __name__=="__main__":
    env = DeltaArrayEnv("COM7")
    # env = DeltaArrayEnv("/dev/tty7")
    # env.move_over_trajectory("vertical")
    # env.planar_translation("up")
    # env.move_delta_array((162.378,131.25))
    # env.move_delta_array((162.378,131.25), "rotate", np.pi/4,zmax=8,zmin=6)
    # env.move_delta_array((162.378,131.25), "down", np.pi/4,zmax=8,zmin=7, wall=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(4,0),(4,1),(5,0),(5,1),(6,0),(6,1),(7,0),(7,1)])
    env.reset()
    # env.glamorous([(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7)], 6)
    # env.glamorous([(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7)], 7)
    # env.glamorous([(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7)], 8)
    # env.glamorous([(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)], 9)
    # env.glamorous([(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7)], 10)
    # env.glamorous([(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7)], 11)
    # env.glamorous([(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7)], 12)
    # env.glamorous([(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)], 13)
    # env.move_delta_array((162.378,0), "converge",zmax=8,zmin=6.5)
    # env.move_delta_array((162.378,131.25), "up")
    # env.grip_and_move_objects()