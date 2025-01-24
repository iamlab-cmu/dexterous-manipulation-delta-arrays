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

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

import utils.nn_helper as nn_helper
from utils.vision_utils import GroundedSAM
import utils.geom_helper as geom_helper

low_z = 9.2
mid_z = 7.5
high_z = 5.5
BUFFER_SIZE = 20

current_frame = None
lock = threading.Lock()

def capture_and_convert():
    global current_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Stream', frame)
        # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with lock:
            current_frame = frame
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

def start_capture_thread():
    capture_thread = threading.Thread(target=capture_and_convert)
    capture_thread.daemon = True
    capture_thread.start()
    
class DeltaArrayReal:
    def __init__(self, config):
        self.max_agents = config['delta_array_size']
        self.lower_green_filter = np.array([30, 5, 5])
        self.upper_green_filter = np.array([90, 255, 255])
        self.img_size = np.array((1080, 1920))
        self.plane_size = np.array([(0.009, -0.376),(0.24200, 0.034)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.nn_helper = nn_helper.NNHelper(self.plane_size, real_or_sim="real")

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
        
        if config['traditional']:
            self.traditional = True
        else:
            self.traditional = False
            self.grounded_sam = GroundedSAM(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
                                segmentation_model="facebook/sam-vit-base",
                                device=config['rl_device'])
        
        self.obj_name = config['obj_name']
        self.MegaTestingLoop = pkl.load(open('./data/test_trajs_real.pkl', 'rb'))
        
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()

    def setup_delta_agents(self):
        self.delta_agents = {}
        # Obtain numbers of 2x2 grids
        for i in range(1, 17):
            # Get IP Addr and socket of each grid and classify them as useful or useless
            # if i!= 10:
            try:
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.05)
                self.delta_agents[i-1] = DeltaArrayAgent(esp01, i)
            except Exception as e:
                print("Error at robot ID: ", i)
                raise e
        self.reset()
        
    def move_robots(self, actions, practicalize=False, reset=False):
        for i, idx in enumerate(self.active_idxs):
            # print(f'Robot {idx} is moving to {actions[i]}')
            if reset:
                traj = [[actions[i][0], actions[i][1], high_z] for _ in range(20)]
                traj = self.practicalize_traj_reset(traj)
            else:
                traj = [[actions[i][0], actions[i][1], low_z] for _ in range(20)]
                if practicalize:
                    traj = self.practicalize_traj(traj)
            self.delta_agents[self.RC.robo_dict_inv[idx] - 1].save_joint_positions(idx, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx])

        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots...")
        self.wait_until_done()
        print("Done!")
    
    def reset(self):
        self.active_idxs.clear()
        self.ep_reward = 0
        self.init_state = np.zeros((self.max_agents, self.state_dim))
        self.final_state = np.zeros((self.max_agents, self.state_dim))
        self.actions = np.zeros((self.max_agents, 2))
        self.pos = np.zeros((self.max_agents, 1))

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
        
    def convert_world_2_pix(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = (vecs[:, 0] - self.plane_size[0][0]) / self.delta_plane_x * 1080
        result[:, 1] = 1920 - (vecs[:, 1] - self.plane_size[0][1]) / self.delta_plane_y * 1920
        return result
    
    def convert_pix_2_world(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = vecs[:, 0] / 1080 * self.delta_plane_x + self.plane_size[0][0]
        result[:, 1] = (1920 - vecs[:, 1]) / 1920 * self.delta_plane_y + self.plane_size[0][1]
        return result
    
    def get_bdpts(self):
        img = current_frame.copy()
        if self.traditional:
            