import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from spatialmath import SE3
import spatialmath as sm
import matplotlib.pyplot as plt
import argparse
import glfw
import cv2
import sys
sys.path.append("..")
from threading import Lock

from config.delta_array_generator import DeltaArrayEnvCreator


class BaseMJEnv:
    def __init__(self, args):
        # Create the environment
        self.obj_name = args['obj_name']
        self.env_creator = DeltaArrayEnvCreator(self.obj_name)
        self.env_creator.create_env(args['num_rope_bodies'])
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(args['path'])
        self.data = mujoco.MjData(self.model)
        self.gui = args['gui']
        self.setup_gui()
        self.gui_lock = Lock()
        
        mujoco.mj_forward(self.model, self.data)

    def setup_gui(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        if self.gui:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'window', width=1920, height=1080)
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen', width=1920, height=1080)
            
        self.viewer.cam.lookat = lookat
        self.viewer.cam.distance = distance
        self.viewer.cam.elevation = elevation
        self.viewer.cam.azimuth = azimuth

    def get_image(self):
        self.rgb_pixels = self.viewer.read_pixels()
        return self.rgb_pixels

    def update_sim(self):
        mujoco.mj_step(self.model, self.data, nstep=self.args['skip'])
        if self.gui:
            self.viewer.render()

    def set_obj_pose(self, body_name, pose):
        self.data.body(self.obj_name).qpos = pose
        mujoco.mj_forward(self.model, self.data)

    def get_obj_pose(self, body_name):
        return self.data.body(self.obj_name).qpos
    
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError