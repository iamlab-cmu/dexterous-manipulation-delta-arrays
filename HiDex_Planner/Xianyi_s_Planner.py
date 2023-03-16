import numpy as np
import pandas as pd
from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
import delta_array_utils.get_coords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm
import socket
import time
from delta_array_utils.control_delta_arrays import DeltaArrayAgent
import delta_array_utils.delta_trajectory_pb2 as delta_trajectory_pb2
from delta_array_utils.serial_robot_mapping import delta_comm_dict, inv_delta_comm_dict

s_p = 1.5 #side length of the platform
s_b = 4.3 #side length of the base
l = 4.5 #length of leg attached to platform
Delta = Prismatic_Delta(s_p, s_b, l)

df = pd.read_csv('./data/cube_ouput.csv', header=None)
df = df.replace(' nan', -7.0)

traj = np.zeros((20, 12))
nan_array = np.array((-7.0,-7.0,-7.0))
nan_array_substitute = np.array((0.0,0.0,5.5))
z_val = 11

default_pos = np.array([[-2.165, 0, 0],
                        [0, 3.75, 0],
                        [-4.3301, 3.75, 0],
                        [-6.4951, 0, 0]])


robot_no = 11
port = 80
timeout = 100
BUFFER_SIZE = 20
esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
esp01.connect((inv_delta_comm_dict[robot_no], port))
esp01.settimeout(0.1)
DDA = DeltaArrayAgent(esp01, robot_no)
def create_joint_positions(val):
    a = []
    for i in range(4):
        for j in range(3):
            a.append(val[j])
    return a

def rotate(vector, angle, plot=False):
        # Rotation from Delta Array axis to cartesian axis = 30 degrees. 
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        vector = vector@rot_matrix
        return vector

def wait_until_done():
    done_moving = False
    while not done_moving:
        try:
            received = esp01.recv(BUFFER_SIZE)
            ret = received.decode().strip()
            if ret == "B":
                done_moving = True
                time.sleep(0.5)
        except Exception as e: 
            # print(e)
            pass
    time.sleep(0.5)
    return 

for n, row in enumerate(df.iterrows()):
    row = row[1].to_numpy(dtype=np.float32)[:-7].reshape((4,6))
    jts = []

    for m, i in enumerate(row[:,:3]):
        if (i == nan_array).all():
            print(nan_array_substitute)
            jts.append(np.clip(np.array(Delta.IK(nan_array_substitute))* 0.01, 0.005, 0.095))
        else:
            body_frame = i - default_pos[m]
            print(i,default_pos[m])
            ee_pos = np.array((body_frame[0],body_frame[1]))
            jts.append(np.clip(np.array(Delta.IK([*rotate(ee_pos, 0),z_val]))* 0.01, 0.005, 0.095))
    # print(jts)
    # new_jts = [jts[2],jts[3],jts[1],jts[0]]
    # new_jts = [jts[1],jts[0],jts[3],jts[2]]
    # traj[5*n:5*(n+1)] = np.hstack(new_jts)
    traj[n] = np.hstack(jts)
    # traj[n] = np.hstack(new_jts)
    # print(np.hstack(jts))
    
    if n%5 == 4:
        traj[n:] = np.hstack(jts)
        # traj[n:] = np.hstack(new_jts)
        # print(traj.shape)
        delta_message = delta_trajectory_pb2.DeltaMessage()
        delta_message.id = robot_no
        delta_message.request_joint_pose = False
        delta_message.request_done_state = False
        delta_message.reset = False

        for j in traj:
            _ = [delta_message.trajectory.append(j[i]) for i in range(12)]
        serialized = delta_message.SerializeToString()
        esp01.send(b'\xa6~~'+ serialized + b'\xa7~~\r\n')
        del delta_message.trajectory[:]
        x = input()
        # break