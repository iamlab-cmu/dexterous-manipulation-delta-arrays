import jax.numpy as jnp
import mujoco
from mujoco import mjx

import numpy as np
import sys
sys.path.append("..")
import mujoco.viewer

from config.delta_array_generator import DeltaArrayEnvCreator

class BaseMJXEnv:
    def __init__(self, config):
        self.env_creator = DeltaArrayEnvCreator(mjx=True)
        env_xml = self.env_creator.create_env("ALL", config['num_rope_bodies'])
        self.config = config

        self.model = mujoco.MjModel.from_xml_string(env_xml)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        
        act_low = jnp.array([*self.model.actuator_ctrlrange[:, 0]])
        act_high = jnp.array([*self.model.actuator_ctrlrange[:, 1]])
        self.act_scale = (act_high - act_low) / 2.0
        self.act_bias = (act_high + act_low) / 2.0
        
        self.gui = config['gui']
        self.setup_camera()
        self.fps_ctr = 0
        
        mujoco.mj_forward(self.model, self.data)
        
    def setup_camera(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        self.renderer = mujoco.Renderer(self.model, 1080, 1920)
        self.renderer.disable_segmentation_rendering()
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = lookat
        self.camera.distance = distance
        self.camera.elevation = elevation
        self.camera.azimuth = azimuth

    def setup_gui(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        self.setup_camera(lookat, distance, elevation, azimuth)
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
            self.fps_ctr += 1
            mujoco.mj_step(self.model, self.data)    
            if (self.fps_ctr % 20 == 0):
                recorder.add_frame(self.get_image())

    def update_sim_gui(self, simlen, recorder=None):
        if self.gui:
            for i in range(simlen):
                mujoco.mj_step(self.model, self.data)
                if recorder is not None:
                    recorder.add_frame(self.get_image())
                self.viewer.sync()
        elif (recorder is not None):
            self.update_sim_recorder(simlen, recorder)
            
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError