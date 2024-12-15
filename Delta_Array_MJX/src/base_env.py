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

    def get_image(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def update_sim(self):
        mujoco.mj_step(self.model, self.data)
        if self.gui:
            self.viewer.sync()
            # time.sleep(0.0001)
            # img = self.get_image()
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Delta Array Window', img)
            # cv2.waitKey(1)

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