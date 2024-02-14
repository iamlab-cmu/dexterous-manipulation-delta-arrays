import numpy as np
import matplotlib.pyplot as plt

import mujoco
import glfw
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('./config/env.xml')
data = mujoco.MjData(model)

glfw.init()
glfw.window_hint(glfw.VISIBLE, 0)
window = glfw.create_window(*glfw.get_video_mode(glfw.get_primary_monitor()).size, "Offscreen", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
cam.lookat = np.array((0.13125, 0.1407285, 1.5))
cam.distance = 0.85
cam.azimuth = 0
cam.elevation = 90
scene = mujoco.MjvScene(model, 10000)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
viewport = mujoco.MjrRect(0, 0, 480, 640)
rgb_pixels = np.zeros((480, 640, 3), dtype=np.uint8)

robot_id = 20
offset_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip_0")

for i in range(10000):
    if i==0:
        model.body_pos[robot_id+offset_id, 2] = 1.02
        data.ctrl[2*robot_id: 2*robot_id+2] = (0.01, 0.01)
    elif (i%100==0):
        data.ctrl[2*robot_id: 2*robot_id+2] = data.ctrl[2*robot_id: 2*robot_id+2]*np.array((-1, -1))
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_readPixels(rgb_pixels, None, viewport, context)
        plt.imshow(rgb_pixels)
        plt.show()
        
    mujoco.mj_step(model, data)