import delta_trajectory_pb2 as delta_trajectory_pb2
import numpy as np
from serial import Serial
from math import *
import time
from Prismatic_Delta import Prismatic_Delta
from get_coords import RoboCoords
import serial_robot_mapping as srm
import socket

NUM_MOTORS = 12
NUM_AGENTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
BUFFER_SIZE = 20
s_p = 1.5 #side length of the platform
s_b = 4.3 #side length of the base
l = 4.5 #length of leg attached to platform
delta_message = delta_trajectory_pb2.DeltaMessage()
delta_message.request_joint_pose = False
delta_message.request_done_state = False
delta_message.reset = False
min_joint_pos = 0.005
max_joint_pos = 0.0985
Delta = Prismatic_Delta(s_p, s_b, l)

##########################################################################
robot_id = 12                                                   # ROBOT ID
##########################################################################
delta_message.id = robot_id
ip_addr = srm.inv_delta_comm_dict[robot_id]
esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
esp01.connect((ip_addr, 80))
esp01.settimeout(0.1)


RC = RoboCoords()
robots = RC.robo_dict[robot_id]
robots_xy = np.array([np.array(RC.robot_positions[robot]) for robot in robots])
com = np.mean(robots_xy, axis=0)

away = RC.get_dist_vec(com, robots_xy, norm=True, angle=0)
print(robots_xy, away)
for j in range(20):
    pts = []
    if j==0:
        for k in -2*away:
            pts.extend(Delta.IK([*k, 10.5]))
    elif j==1:
        for k in 2*away:
            pts.extend(Delta.IK([*k, 10.5]))
    elif j<10.5:
        for k in RC.rotate(2*away, np.pi/4):
            pts.extend(Delta.IK([*k, 7]))
    elif j<14:
        for k in RC.rotate(2*away, 0):
            pts.extend(Delta.IK([*k, 7]))
    elif j<18:
        for k in RC.rotate(2*away, -np.pi/4):
            pts.extend(Delta.IK([*k, 7]))
    else:
        for k in -2*away:
            pts.extend(Delta.IK([*k, 10.5]))


    pts = np.array(pts) * 0.01
    pts = np.clip(pts,0.001,0.095)
    _ = [delta_message.trajectory.append(pts[i]) for i in range(12)]



    # if j<10:
    #     ee_pts = [x[j],y[j],5.0]
    # else:
    #     ee_pts = [x[j],y[j],5.0]
    # # ee_pts = [0,0,12]
    # pts = Delta.IK(ee_pts)
    # pts = np.array(pts) * 0.01
    # pts = np.clip(pts,0.001,0.095)
    # _ = [delta_message.trajectory.append(pts[i%3]) for i in range(12)]

serialized = delta_message.SerializeToString()
esp01.send(b'\xa6~~'+ serialized + b'\xa7~~\r\n')
del delta_message.trajectory[:]