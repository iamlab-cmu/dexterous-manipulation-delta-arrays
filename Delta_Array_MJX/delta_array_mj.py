import numpy as np
import matplotlib.pyplot as plt
import config.delta_array_generator

import mujoco
import glfw
import mujoco_viewer

class DeltaArrayMJ:
    def __init__(self, mjcf_path, cfg):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg

        self.setup_cam(cfg)
    
    def setup_cam(self):
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, self.cfg['visible'])
        window = glfw.create_window(self.cfg['width'], self.cfg['height'], self.cfg["title"], None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)

        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = np.array((0.13125, 0.1407285, 1.5))
        self.cam.distance = 0.85
        self.cam.azimuth = 0
        self.cam.elevation = 90
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()

        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        self.rgb_pixels = np.zeros((height, width, 3), dtype=np.uint8)

    def get_image(self):
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_pixels, None, self.viewport, self.context)
        self.rgb_pixels = np.flipud(self.rgb_pixels)







if __name__ == "__main__":
    mjcf_path = './config/env.xml'
    cfg = {
        "width": 1920,
        "height": 1080,
        "visible": 0,
        "title": "no_title"
    }
    delta_array_mj = DeltaArrayMJ(mjcf_path, cfg)
    print("done")