import numpy as np
import matplotlib.pyplot as plt
import config.delta_array_generator

import mujoco
import glfw
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('./config/env.xml')
data = mujoco.MjData(model)

width, height = 1920, 1080

glfw.init()
glfw.window_hint(glfw.VISIBLE, 1)
window = glfw.create_window(width, height, "window", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)

opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
cam.lookat = np.array((0.13125, 0.1407285, 1.5))
# cam.fovy = 42.1875
cam.distance = 0.85
cam.azimuth = 0
cam.elevation = 90
scene = mujoco.MjvScene(model, maxgeom=10000)
pert = mujoco.MjvPerturb()


context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
rgb_pixels = np.zeros((height, width, 3), dtype=np.uint8)

robot_id = 20
offset_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip_0")

for i in range(10000):
    if i==0:
        model.body_pos[robot_id+offset_id, 2] = 1.02
        model.body_pos[0+offset_id, 2] = 1.02
        model.body_pos[63+offset_id, 2] = 1.02
        data.ctrl[2*robot_id: 2*robot_id+2] = (0.01, 0.01)
    elif (i%100==0):
        data.ctrl[2*robot_id: 2*robot_id+2] = data.ctrl[2*robot_id: 2*robot_id+2]*np.array((-1, -1))
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        # mujoco.mjr_readPixels(rgb_pixels, None, viewport, context)
        # plt.imshow(np.flipud(rgb_pixels))
        # plt.show()
        
    mujoco.mj_step(model, data)