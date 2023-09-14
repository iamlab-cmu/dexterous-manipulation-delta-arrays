import delta_trajectory_pb2 as delta_trajectory_pb2
import numpy as np
from serial import Serial
from math import *
import time
from Prismatic_Delta import Prismatic_Delta
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


angles = np.linspace(0, 0*np.pi, 20)
x = 0 * np.cos(angles)
y = 0 * np.sin(angles)
for j in range(20):
    ee_pts = [x[j],y[j],5.0]
    # ee_pts = [0,0,12]
    pts = Delta.IK(ee_pts)
    pts = np.array(pts) * 0.01
    pts = np.clip(pts,0.001,0.095)
    _ = [delta_message.trajectory.append(pts[i%3]) for i in range(12)]

serialized = delta_message.SerializeToString()
esp01.send(b'\xa6~~'+ serialized + b'\xa7~~\r\n')
del delta_message.trajectory[:]