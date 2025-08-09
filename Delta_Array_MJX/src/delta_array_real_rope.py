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
import math

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

import utils.nn_helper as nn_helper
from utils.vision_utils import VisUtils
import utils.geom_helper as geom_helper
from utils.video_utils import VideoRecorder
from utils.visualizer_utils import Visualizer
import utils.rope_utils as rope_utils

LOW_Z = 9.8
MID_Z = 7.5
HIGH_Z = 5.5
BUFFER_SIZE = 20
theta = -np.pi / 2
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,             0,              1]
])

current_frame = None
global_bd_pts = None
# lock = threading.Lock()

def capture_and_convert(stop_event, lock, rl_device, trad, plane_size, save_vid, vid_name=None):
    global current_frame, global_bd_pts
    
    vis_utils = VisUtils(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
                        segmentation_model="facebook/sam-vit-base",
                        device=rl_device,
                        traditional=trad, plane_size=plane_size)
    vis_utils.set_label("green rope")
    
    video_recorder = None
    if save_vid:
        video_recorder = VideoRecorder(output_dir="./data/videos/rope", fps=120, resolution=(1920, 1080))
        video_recorder.start_recording(vid_name)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # bd_pts_world = vis_utils.get_bdpts(frame, n_obj)
        
        bd_pts = vis_utils.get_bd_pts(frame)

        with lock:
            global_bd_pts = bd_pts.copy()
            current_frame = frame.copy()
            
        if current_frame is not None:
            cv2.imshow('Stream', current_frame)

        if video_recorder is not None:
            video_recorder.add_frame(current_frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if video_recorder is not None:
        video_recorder.stop_recording()
    print("[Child] Exiting child process safely.")
    
def start_capture_thread(lock, rl_device, trad, plane_size, save_vid, vid_name=None):
    # Create an Event to signal the capture thread to stop
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_and_convert,
        args=(
            stop_event,
            lock,
            rl_device,
            trad,
            plane_size,
            save_vid,
            vid_name,
        ),
        daemon=True  # so it won’t block your main thread from exiting if you forget to join
    )
    capture_thread.start()
    return stop_event, capture_thread
    
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def delta_angle(yaw1, yaw2):
    return abs(np.arctan2(np.sin(yaw1 - yaw2), np.cos(yaw1 - yaw2)))
    
class DeltaArrayRealRope:
    def __init__(self, config):
        self.max_agents = 64
        self.img_size = np.array((1080, 1920))
        # self.plane_size = np.array([(0.009, -0.376),(0.24200, 0.034)])
        self.plane_size = np.array([(0.009,  -0.034),(0.24200, 0.376)])
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
        # self.all_robots = np.stack(np.meshgrid(np.arange(8), np.arange(8)), 2).reshape(-1, 2)
        self.all_robots = np.arange(64)
        self.rope = True
        self.reward_scale = config['reward_scale']
        
        
        # goal_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(global_mask, mask=True))
        # self.goal_bd_pts = rope_utils.sample_points(goal_rope_coords, 50)
        self.goal_bd_pts = pkl.load(open("data/temp/rope_goal_coords.pkl", "rb"))
        
        self.current_boundary_sampled_pts = None # World coordinates
        self.active_idxs = np.array([], dtype=int) # Indices of active robots (0-63)
        self.init_nn_boundary_pts = np.empty((0, 2)) # World coords of boundary points closest to active robots
        self.boundary_indices = np.array([], dtype=int) # Indices into the sampled boundary array (0-49)
        self.goal_nn_boundary_pts = np.empty((0, 2)) # World coords of corresponding goal boundary points
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()
        self.config = config
        
        # self.visualizer = Visualizer()
        
    def convert_pix_2_world(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = vecs[:, 0] / 1080 * self.delta_plane_x + self.plane_size[0][0]
        result[:, 1] = (1920 - vecs[:, 1]) / 1920 * self.delta_plane_y + self.plane_size[0][1]
        return result

    def reset(self):
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0

        self.current_boundary_sampled_pts = None 
        self.active_idxs = np.array([], dtype=int) 
        self.init_nn_boundary_pts = np.empty((0, 2))
        self.boundary_indices = np.array([], dtype=int) 
        self.goal_nn_boundary_pts = np.empty((0, 2))
        self.n_idxs = 0
        
        for i in set(self.RC.robo_dict_inv.values()):
            self.delta_agents[i-1].reset()
            self.to_be_moved.append(self.delta_agents[i-1])

        # print("Resetting Delta Robots...")
        self.wait_until_done()
        
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
        
    def reconnect_delta_agents(self):
        for i in range(1, 17):
            self.delta_agents[i-1].esp01.close()
            del self.delta_agents[i-1]
            
        self.delta_agents = {}
        for i in range(1, 17):
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
        
    def practicalize_traj(self, traj):
        pass
        
    def move_robots(self, active_idxs, actions, z_level, from_NN=False, practicalize=False):
        actions = self.clip_actions_to_ws(95*actions.copy())
        for i, idx in enumerate(active_idxs):
            y_mult = -1 # if from_NN else -1
            traj = [[actions[i][0], y_mult*actions[i][1], z_level] for _ in range(20)]
            if practicalize:
                traj = self.practicalize_traj(traj)
            idx2 = (idx//8, idx%8)
            self.delta_agents[self.RC.robo_dict_inv[idx2] - 1].save_joint_positions(idx2, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx2])

        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        # print("Moving Delta Robots...")
        self.wait_until_done()
        # print("Done!")

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
    
    def set_z_positions(self, active_idxs=None, low=True):
        # TODO: Convert active idxs from 1x 0 - 63 to 2x 0-7. (idx//2, idx%2)
        if active_idxs is None:
            active_idxs = self.all_robots.copy()
        actions = []
        for idx in active_idxs:
            actions.append([0, 0])
        self.move_robots(active_idxs, np.array(actions), LOW_Z if low else HIGH_Z, practicalize=False)
        
    def clip_actions_to_ws(self, actions):
        return self.Delta.clip_points_to_workspace(actions)
    
    def vs_action(self, act_grasp, random=False):
        if random:
            actions = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2))
        else:
            self.sf_bd_pts, self.sf_nn_bd_pts = self.get_current_bd_pts()
            displacement_vectors = self.goal_nn_bd_pts - self.sf_nn_bd_pts
            actions = act_grasp + displacement_vectors
        return actions
    
    def get_bdpts(self):
        rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(global_bd_pts, mask=True))
        bd_pts = rope_utils.sample_points(rope_coords, total_points=50)
        # plt.scatter(bd_pts[:, 0], bd_pts[:, 1], color='red')
        # plt.show()
        return bd_pts
    
    def set_rl_states(self, actions=None, final=False, test_traj=False):
        if final:
            self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
            if not self.rope:
                self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
                self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
                self.final_state[:self.n_idxs, 4:6] = actions[:self.n_idxs]
            else:
                self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
                self.final_state[:self.n_idxs, 4:6] = actions[:self.n_idxs]
        else:
            
            self.n_idxs = len(self.active_idxs)
            self.pos[:self.n_idxs] = self.active_idxs.copy()
            self.raw_rb_pos = self.nn_helper.kdtree_positions_world[self.active_idxs]
            
            self.init_state[:self.n_idxs, :2] = self.init_nn_bd_pts - self.raw_rb_pos
            self.init_state[:self.n_idxs, 2:4] = self.goal_nn_bd_pts - self.raw_rb_pos
            
            acts = 0.8*self.clip_actions_to_ws(self.init_nn_bd_pts - self.raw_rb_pos)
            self.init_state[:self.n_idxs, 4:6] = acts
            self.final_state[:self.n_idxs, 2:4] = self.init_state[:self.n_idxs, 2:4].copy()
            
            return acts.copy()
            # plt.scatter(self.init_state[:self.n_idxs, 0], self.init_state[:self.n_idxs, 1], color='red')
            # plt.scatter(self.init_state[:self.n_idxs, 2], self.init_state[:self.n_idxs, 3], color='blue')
            # plt.quiver(self.init_state[:self.n_idxs, 0], self.init_state[:self.n_idxs, 1], self.init_state[:self.n_idxs, 4], self.init_state[:self.n_idxs, 5], color='green')
            # plt.show()
            
        
    def get_current_bd_pts(self):
        # current_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(global_bd_pts, mask=True))
        # current_bd_pts = rope_utils.get_aligned_smol_rope(current_rope_coords, self.goal_bd_pts.copy(), N=50)
        current_bd_pts = global_bd_pts.copy()
        current_nn_bd_pts = current_bd_pts[self.bd_idxs]
        return current_bd_pts, current_nn_bd_pts
        
    def set_goal_nn_bd_pts(self):
        self.goal_nn_bd_pts = self.goal_bd_pts[self.bd_idxs].copy()
    
    def get_active_idxs(self):
        self.active_idxs, self.init_nn_bd_pts, self.bd_idxs = self.nn_helper.get_nn_robots_rope(self.init_bd_pts)
        if len(self.init_nn_bd_pts) == 0:
            return False
        return True
        
    def compute_reward(self, actions):
        init_dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.init_nn_bd_pts, axis=1))
        final_dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        delta = 10*(init_dist - final_dist)
        return delta/10, delta/self.reward_scale
    
    def soft_reset(self):
        self.set_z_positions(self.active_idxs, low=False)
        # if goal_2Dpose is None:
        #     self.set_z_positions(self.active_idxs, low=False)
        # else:
        #     self.reconnect_delta_agents()
            
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0
        
        
        
        self.init_bd_pts = self.get_bdpts()
        self.get_active_idxs()
        self.set_goal_nn_bd_pts()
        act_grasp = self.set_rl_states()
        return act_grasp
        
    def rollout(self, act_grasp, actions, from_NN):
        if actions.shape[-1] == 3:
            self.final_state[:self.n_idxs, 4:6] = actions[:, :2]
            active_idxs = np.array(self.active_idxs)
            sel_idxs = actions[:, 2] < 0
            # inactive_idxs = active_idxs[actions[:, 2] > 0]
            active_idxs = active_idxs[sel_idxs]
            # if len(inactive_idxs) > 0:
            #     self.set_z_positions(active_idxs=list(inactive_idxs), low=False)  # Set to high
            execute_actions = actions[sel_idxs, :2]
        else:
            active_idxs = np.array(self.active_idxs)
            sel_idxs = np.ones_like(actions[:, 0], dtype=bool)
            execute_actions = actions.copy()
                
        self.set_z_positions(active_idxs, low=True)
        self.move_robots(active_idxs, act_grasp[sel_idxs], LOW_Z, practicalize=False)
                
        self.move_robots(active_idxs, execute_actions, LOW_Z, from_NN)
        self.set_rl_states(actions[:, :2], final=True, test_traj=True)
        dist, reward = self.compute_reward(actions)
        return dist, reward