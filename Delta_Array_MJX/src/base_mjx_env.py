import jax.numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx

import numpy as np
import sys
sys.path.append("..")
import mujoco.viewer

from config.delta_array_generator import DeltaArrayEnvCreator

class BaseMJXEnv(eqx.Module):
    # --- JAX-compatible PyTree fields ---
    mjx_model: mjx.Model
    act_scale: jnp.ndarray
    act_bias: jnp.ndarray

    model: mujoco.MjModel = eqx.field(static=True)
    data: mujoco.MjData = eqx.field(static=True)
    config: dict = eqx.field(static=True)
    gui: bool = eqx.field(static=True)
    renderer: mujoco.Renderer = eqx.field(static=True, default=None)
    camera: mujoco.MjvCamera = eqx.field(static=True, default=None)
    viewer: mujoco.viewer = eqx.field(static=True, default=None)

    def __init__(self, config):
        env_creator = DeltaArrayEnvCreator(mjx=True)
        env_xml = env_creator.create_env("ALL", config['num_rope_bodies'])
        
        # --- Assign static fields first ---
        self.config = config
        self.gui = config['gui']
        self.model = mujoco.MjModel.from_xml_string(env_xml)
        self.data = mujoco.MjData(self.model)

        # --- Assign JAX-compatible fields ---
        self.mjx_model = mjx.put_model(self.model)
        
        act_low = jnp.array([*self.model.actuator_ctrlrange[:, 0]])
        act_high = jnp.array([*self.model.actuator_ctrlrange[:, 1]])
        self.act_scale = (act_high - act_low) / 2.0
        self.act_bias = (act_high + act_low) / 2.0
        
        # --- Visualization setup (non-JAX objects) ---
        self.renderer, self.camera = self.setup_camera()
        if self.gui:
            self.viewer = self.setup_gui()
        
        mujoco.mj_forward(self.model, self.data)
        
    def setup_camera(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        renderer = mujoco.Renderer(self.model, 1080, 1920)
        renderer.disable_segmentation_rendering()
        camera = mujoco.MjvCamera()
        camera.lookat = lookat
        camera.distance = distance
        camera.elevation = elevation
        camera.azimuth = azimuth
        return renderer, camera

    def setup_gui(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        # This method is for visualization and should not be JIT-compiled.
        # It creates and returns a viewer, which we store in a static field.
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        viewer.cam.lookat = lookat
        viewer.cam.distance = distance
        viewer.cam.elevation = elevation
        viewer.cam.azimuth = azimuth
        return viewer

    def get_segmentation(self, target_id=67):
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        seg = self.renderer.render()
        geom_ids = seg[:, :, 0]
        mask = (geom_ids == target_id)
        return (255*mask).astype(np.uint8)

    def get_image(self):
        # Note: self.renderer and self.data are static fields, so this method is not JIT-able.
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