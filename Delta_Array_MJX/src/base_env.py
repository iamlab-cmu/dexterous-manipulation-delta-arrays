import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from spatialmath import SE3
import spatialmath as sm
import matplotlib.pyplot as plt
import argparse
import cv2
import sys
import time
sys.path.append("..")
from threading import Lock
import mujoco.viewer

import utils.visualizer_utils as visualizer_utils
from config.delta_array_generator import DeltaArrayEnvCreator


class BaseMJEnv:
    def __init__(self, args, obj_name):
        # Create the environment
        self.obj_name = obj_name
        self.env_creator = DeltaArrayEnvCreator()
        env_xml = self.env_creator.create_env(self.obj_name, args['num_rope_bodies'])
        self.args = args

        self.model = mujoco.MjModel.from_xml_string(env_xml)
        self.data = mujoco.MjData(self.model)
        self.gui = args['gui']
        self.setup_gui()
        self.gui_lock = Lock()
        self.visualizer = visualizer_utils.StateActionVisualizer()
        
        mujoco.mj_forward(self.model, self.data)

    def setup_gui(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        self.width, self.height = 1920, 1080
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.disable_segmentation_rendering()
        self.camera = mujoco.MjvCamera()
            
        self.camera.lookat = lookat
        self.camera.distance = distance
        self.camera.elevation = elevation
        self.camera.azimuth = azimuth
        
        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat = lookat
            self.viewer.cam.distance = distance
            self.viewer.cam.elevation = elevation
            self.viewer.cam.azimuth = azimuth

    def get_segmentation(self, target_id=67):
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        seg = self.renderer.render()
        geom_ids = seg[:, :, 0]
        mask = (geom_ids == target_id)
        return (255*mask).astype(np.uint8)

    def get_image(self):
        self.renderer.update_scene(self.data, camera=self.camera)        
        return self.renderer.render()
    
    def update_sim_recorder(self, simlen, recorder):
        for i in range(simlen):
            mujoco.mj_step(self.model, self.data)    
            recorder.add_frame(self.get_image())

    def update_sim(self, simlen, td=None, recorder=None):
        if self.gui:
            for i in range(simlen):
                mujoco.mj_step(self.model, self.data)
                if recorder is not None:
                    recorder.add_frame(self.get_image())
                self.viewer.sync()
                if td is None:
                    time.sleep(0.0005)
                else:
                    time.sleep(td)
        else:
            mujoco.mj_step(self.model, self.data, simlen)

    def set_obj_pose(self, body_name, pose):
        self.data.body(self.obj_name).qpos = pose
        mujoco.mj_forward(self.model, self.data)

    def get_obj_pose(self, body_name):
        return self.data.body(self.obj_name).qpos
    
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
    
    def visualize_image(self, acts, block_pos, init, goal):
        fig = self.visualizer.visualize_state_and_actions(
            robot_positions=self.nn_helper.kdtree_positions_world[self.active_idxs],
            init_state=self.init_state,
            actions=acts,
            block_position=block_pos,
            n_idxs=self.n_idxs,
            all_robot_positions=self.nn_helper.kdtree_positions_world,  # All possible positions
            active_idxs=self.active_idxs,
            init_bd_pts=init,
            goal_bd_pts=goal
        )
        plt.show()
        
    
    def plot_visual_servo_debug(self, init_bd_pts, goal_bd_pts, semi_final_bd_pts, final_bd_pts, actions, title="Visual Servo Debug"):
        """
        Plot boundary points and actions for visual servoing debug
        
        Args:
            init_bd_pts: Initial boundary points (N, 2)
            goal_bd_pts: Goal boundary points (N, 2)
            semi_final_bd_pts: Semi-final boundary points (N, 2)
            actions: Action vectors (N, 2)
        """
        plt.figure(figsize=(12, 9))
        r_poses = self.nn_helper.kdtree_positions_world

        # Plot 1: All boundary points
        plt.subplot(111)
        plt.scatter(r_poses[:, 1], r_poses[:, 0], c='#888888ff')
        plt.scatter(self.init_bd_pts[:, 1], self.init_bd_pts[:, 0], c='#E69F00', label='Init_big', alpha=0.5)
        plt.scatter(self.goal_bd_pts[:, 1], self.goal_bd_pts[:, 0], c='#56B4E9', label='Goal_big', alpha=0.5)
        plt.scatter(self.final_bd_pts[:, 1], self.final_bd_pts[:, 0], c='#009E73', label='Goal_big', alpha=0.5)
        plt.scatter(init_bd_pts[:, 1], init_bd_pts[:, 0], c='#E69F00', label='Initial', alpha=1)
        plt.scatter(goal_bd_pts[:, 1], goal_bd_pts[:, 0], c='orange', label='Goal', alpha=1)
        plt.scatter(semi_final_bd_pts[:, 1], semi_final_bd_pts[:, 0], c='green', label='Semi-final', alpha=1)
        plt.scatter(final_bd_pts[:, 1], final_bd_pts[:, 0], c='blue', label='Final', alpha=1)
        
        act2 = goal_bd_pts - semi_final_bd_pts
        plt.quiver(semi_final_bd_pts[:, 1], semi_final_bd_pts[:, 0], 
                act2[:, 1], act2[:, 0], color='#0072B233', alpha=0.2)
        
        plt.quiver(semi_final_bd_pts[:, 1], semi_final_bd_pts[:, 0], 
                actions[:, 1], actions[:, 0], 
                angles='xy', scale_units='xy', scale=1, 
                color='#D55E0033', alpha=0.2)
        
        # Draw lines between corresponding points
        # for i in range(len(init_bd_pts)):
        #     plt.plot([semi_final_bd_pts[i, 1], goal_bd_pts[i, 1]], 
        #             [semi_final_bd_pts[i, 0], goal_bd_pts[i, 0]], 
        #             'gray', alpha=0.2)
        
        plt.title("Boundary Points")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()