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
global_bd_pts, global_yaw, global_com = None, None, None
# lock = threading.Lock()

def capture_and_convert(stop_event, current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None):
    global current_frame, global_bd_pts, global_yaw, global_com
    
    vis_utils = VisUtils(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
                        segmentation_model="facebook/sam-vit-base",
                        device=rl_device,
                        traditional=trad, plane_size=plane_size)
    
    # traj_world = np.array(current_traj, dtype=np.float32)
    # traj_pixels = vis_utils.convert_world_2_pix(traj_world)
    # n_points = len(traj_pixels)
    # indices = np.linspace(0, 1, n_points)

    # # Color gradient from yellow (BGR: 0,255,255) to deep red (BGR: 0,0,220)
    # colors = np.column_stack([
    #     np.zeros(n_points),  # Blue channel
    #     255 * (1 - indices),  # Green channel
    #     255 * (1 - 0.35 * indices)  # Red channel
    # ]).astype(np.uint8)
    
    # for radius, alpha in zip([8, 6, 4], [0.2, 0.4, 1.0]):
    #     blended_color = (colors * alpha).astype(np.uint8)
    #     cv2.polylines(frame, [traj_pixels], isClosed=False, color=(0, 150, 255), 
    #                 thickness=radius*2, lineType=cv2.LINE_AA)
    #     [cv2.circle(frame, tuple(p), radius, tuple(color.tolist()), -1)
    #     for p, color in zip(traj_pixels, blended_color)]
    
    
    video_recorder = None
    if save_vid:
        video_recorder = VideoRecorder(output_dir="./data/videos/real", fps=30, resolution=(1920, 1080))
        video_recorder.start_recording(vid_name)

    
    camera_matrix = np.load("./utils/calibration_data/camera_matrix.npy") 
    dist_coeffs = np.load("./utils/calibration_data/dist_coeffs.npy")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_size = 0.015

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        bd_pts_world = vis_utils.get_bdpts(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            rvec = rvecs[0, 0, :]
            # tvec = tvecs[0, 0, :]
            R, _ = cv2.Rodrigues(rvec)
            yaw0 = math.atan2(R[1, 0], R[0, 0])
            R =  R @ R_z
            yaw = math.atan2(R[1, 0], R[0, 0])
            
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            marker_corners_2d = corners[0][0]
            com_pix = np.mean(marker_corners_2d, axis=0, dtype=int)
            
            # line_length = 100
            # arrow_x = int(com_pix[0] + line_length * math.cos(yaw0 - np.pi/2))
            # arrow_y = int(com_pix[1] + line_length * math.sin(yaw0 - np.pi/2))
            # cv2.arrowedLine( frame, [*com_pix], (arrow_x, arrow_y), color=(0, 0, 255), thickness=3, tipLength=0.2)
            
            com_world = np.mean(bd_pts_world, axis=0)
            # cv2.polylines(frame, [traj_pixels], isClosed=False, color=(0, 100, 255), 
            #                                     thickness=2, lineType=cv2.LINE_AA)
            # cv2.arrowedLine(frame, tuple(traj_pixels[-2]), tuple(traj_pixels[-1]), 
            #        (0, 0, 255), 2, tipLength=0.3, line_type=cv2.LINE_AA)
            # for i in range(len(traj_pixels)):
            #     px, py = traj_pixels[i]
            #     cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 255), -1)
            
            with lock:
                global_com = com_world
                global_yaw = yaw
                global_bd_pts = bd_pts_world

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
    
def start_capture_thread(current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None):
    # Create an Event to signal the capture thread to stop
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_and_convert,
        args=(
            stop_event,
            current_traj,
            lock,
            rl_device,
            trad,
            plane_size,
            save_vid,
            vid_name
        ),
        daemon=True  # so it won’t block your main thread from exiting if you forget to join
    )
    capture_thread.start()
    return stop_event, capture_thread
    
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def delta_angle(yaw1, yaw2):
    return abs(np.arctan2(np.sin(yaw1 - yaw2), np.cos(yaw1 - yaw2)))
    
class DeltaArrayReal:
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
        
        self.traditional = config['traditional']
        
        self.obj_name = config['obj_name']
        if self.obj_name == "rope":
            self.rope = True
        else:
            self.rope = False
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()
        self.config = config
        
        # self.visualizer = Visualizer()

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
        
    def move_robots(self, active_idxs, actions, z_level, from_NN=False, practicalize=False):
        print(from_NN, actions.shape)
        
        actions = self.clip_actions_to_ws(100*actions.copy())
        for i, idx in enumerate(active_idxs):
            y_mult = -1 # if from_NN else -1
            traj = [[actions[i][0], y_mult*actions[i][1], z_level] for _ in range(20)]
            # if practicalize:
            #     traj = self.practicalize_traj(traj)
            idx2 = (idx//8, idx%8)
            self.delta_agents[self.RC.robo_dict_inv[idx2] - 1].save_joint_positions(idx2, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx2])

        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        print("Moving Delta Robots...")
        self.wait_until_done()
        print("Done!")

    def reset(self):
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0

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
    
    def vs_action(self, random=False):
        if random:
            actions = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2))
        else:
            self.sf_bd_pts, self.sf_nn_bd_pts = self.get_current_bd_pts()
            displacement_vectors = self.goal_nn_bd_pts - self.sf_nn_bd_pts
            actions = self.actions_grasp[:self.n_idxs] + displacement_vectors
        return actions
    
    def set_rl_states(self, actions=None, final=False, test_traj=False):
        if final:
            bdpts, yaw, xy = self.get_bdpts_and_pose()
            self.final_qpos = [*xy, yaw]
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
            
            acts = self.clip_actions_to_ws(self.init_nn_bd_pts - self.raw_rb_pos)
            self.init_state[:self.n_idxs, 4:6] = acts
            self.actions_grasp[:self.n_idxs] = acts
            
            self.final_state[:self.n_idxs, 2:4] = self.init_state[:self.n_idxs, 2:4].copy()
            self.set_z_positions(self.active_idxs, low=True)
            
            # plt.scatter(self.init_state[:self.n_idxs, 0], self.init_state[:self.n_idxs, 1], color='red')
            # plt.scatter(self.init_state[:self.n_idxs, 2], self.init_state[:self.n_idxs, 3], color='blue')
            # plt.quiver(self.init_state[:self.n_idxs, 0], self.init_state[:self.n_idxs, 1], self.init_state[:self.n_idxs, 4], self.init_state[:self.n_idxs, 5], color='green')
            # plt.show()
            
        
    def get_current_bd_pts(self):
        bd_pts = None
        while bd_pts is None:
            bd_pts = global_bd_pts.copy()
        nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), bd_pts, self.init_nn_bd_pts.copy())
        return bd_pts, nn_bd_pts
        
    def get_bdpts_and_pose(self):
        # if self.traditional:
        bd_pts, yaw, com = global_bd_pts.copy(), global_yaw, global_com.copy()
        # else:
        #     bd_pts, yaw, com = self.vis_utils.get_bdpts_and_pose(current_frame.copy())
        return bd_pts, yaw, com
        
    def set_goal_nn_bd_pts(self):
        self.goal_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), self.goal_bd_pts.copy(), self.init_nn_bd_pts.copy())
    
    def get_active_idxs(self):
        idxs, self.init_nn_bd_pts, _ = self.nn_helper.get_nn_robots_objs(self.init_bd_pts, world=True)
        self.active_idxs = list(idxs)
        
    def compute_reward(self, actions):
        # self.visualizer.vis_bd_points(self.init_bd_pts, self.init_nn_bd_pts, self.final_nn_bd_pts, self.goal_nn_bd_pts, self.final_bd_pts, self.goal_bd_pts, actions, self.active_idxs, self.nn_helper.kdtree_positions_world)
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        # ep_reward = np.clip(self.scaling_factor / (dist**2 + self.epsilon), 0, self.max_reward)
        ep_reward = 1 / (10000 * dist**3 + 0.01)
        return dist, ep_reward
    
    def soft_reset(self, init_2Dpose=None, goal_2Dpose=None):
        if goal_2Dpose is None:
            self.set_z_positions(self.active_idxs, low=False)
        else:
            self.reconnect_delta_agents()
            
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0
        
        # First part needs human intervention to align objects with initial pose
        if init_2Dpose is not None:
            self.init_2Dpose = init_2Dpose
            delta_x = 99
            delta_y = 99
            delta_yaw = 99
            while (delta_x > 0.006) or (delta_y > 0.006) or (delta_yaw > 0.1):
                self.init_bd_pts, yaw, com = self.get_bdpts_and_pose()
                delta_x = abs(com[0] - init_2Dpose[0])
                delta_y = abs(com[1] - init_2Dpose[1])
                delta_yaw = delta_angle(yaw, init_2Dpose[2])
                time.sleep(0.1)
                print(f"x_err: {delta_x}; y_err: {delta_y}; yaw_err: {delta_yaw}")
                
        self.init_bd_pts, yaw, com = self.get_bdpts_and_pose()
        
        self.init_qpos = [*com, yaw]
        self.get_active_idxs()
        
        if goal_2Dpose is not None:
            self.goal_bd_pts = geom_helper.get_tfed_2Dpts(self.init_bd_pts, self.init_qpos, goal_2Dpose)
        
        self.set_goal_nn_bd_pts()
        self.set_rl_states()
        
        # print("Grasp Action: ", self.actions_grasp[:self.n_idxs])
        self.move_robots(self.active_idxs, self.actions_grasp[:self.n_idxs], LOW_Z, practicalize=False)