import numpy as np

import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('./config/env.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)
viewer.cam.lookat = np.array((0.13125, 0.1407285, 1.5))
viewer.cam.distance = 0.85
viewer.cam.azimuth = 0
viewer.cam.elevation = 90

robot_id = 0
print(data.qpos[robot_id*7:robot_id*7+7])

for i in range(10000):
    if i%2==0:
        data.qpos[:3] = (0.01, 0.01, 1.02)
    if i%2!=0:
        data.qpos[:3] = (-0.01, -0.01, 1.02)
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()