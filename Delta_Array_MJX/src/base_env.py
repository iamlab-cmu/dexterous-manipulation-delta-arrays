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
        self.visualizer = visualizer_utils.Visualizer()
        
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

    def update_sim(self, simlen, recorder=None):
        if self.gui:
            for i in range(simlen):
                mujoco.mj_step(self.model, self.data)
                if recorder is not None:
                    recorder.add_frame(self.get_image())
                self.viewer.sync()
                time.sleep(0.00075)
        elif recorder is not None:
            self.update_sim_recorder(simlen, recorder)
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