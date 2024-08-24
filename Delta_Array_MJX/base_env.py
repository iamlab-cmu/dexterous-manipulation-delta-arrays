import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from spatialmath import SE3
import spatialmath as sm
import argparse
import glfw
import sys
sys.path.append("..")
from threading import Lock

from config.delta_array_generator import DeltaArrayEnvCreator

gui_lock = Lock()

class BaseMJEnv:
    def __init__(self, args):
        # Create the environment
        self.obj_name = args.obj_name
        self.env_creator = DeltaArrayEnvCreator(self.obj_name)
        self.env_creator.create_env()
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(args.path)
        self.data = mujoco.MjData(self.model)
        self.gui = args.gui
        self.setup_gui()
        self.rgb_pixels = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

    def setup_gui(self, lookat=np.array((0.13125, 0.1407285, 1.5)), distance=0.85, elevation=90, azimuth=0):
        glfw.init()
        if self.gui:
            glfw.window_hint(glfw.VISIBLE, 1)
        else:
            glfw.window_hint(glfw.VISIBLE, 0)

        self.window = glfw.create_window(self.args.width, self.args.height, "window", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = lookat
        self.cam.distance = distance
        self.cam.elevation = elevation
        self.cam.azimuth = azimuth
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

    def get_image(self):
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_pixels, None, self.viewport, self.context)
        self.rgb_pixels = np.flipud(self.rgb_pixels)
        return self.rgb_pixels

    def update_sim(self):
        mujoco.mj_step(self.model, self.data, nstep=self.args.skip)
        if self.gui:
            with gui_lock:
                mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
                mujoco.mjr_render(self.viewport, self.scene, self.context)
                glfw.swap_buffers(self.window)
            glfw.poll_events()

    def set_obj_pose(self, body_name, pose):
        self.data.body(self.obj_name).qpos = pose
        mujoco.mj_forward(self.model, self.data)

    def get_obj_pose(self, body_name):
        return self.data.body(self.obj_name).qpos
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        raise NotImplementedError