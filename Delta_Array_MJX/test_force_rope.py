import mujoco
import mujoco_viewer
import numpy as np
import cv2
import time

model = mujoco.MjModel.from_xml_path('./config/testing_env.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data, 'window')

PRE_SUBSTEPS = 100
POST_SUBSTEPS = 250

body_ids = np.arange(model.body('B_first').id, model.body('B_last').id + 1)

def apply_random_force(data):
    """Apply random force with better scaling"""
    random_id = np.random.choice(body_ids)
    Fmax_xy = 5
    Fmax_z = 35
    fx = np.random.uniform(-Fmax_xy, Fmax_xy) * 100
    fy = np.random.uniform(-Fmax_xy, Fmax_xy) * 100
    fz = np.random.uniform(0, Fmax_z) * 100
    
    # Scale forces based on mass
    mass = model.body_mass[random_id]
    force_scaling = mass * 9.81  # Scale relative to weight
    
    force = np.array([fx, fy, fz, 0, 0, 0]) * force_scaling
    data.xfrc_applied[random_id] = force

# simulate and render
for t in range(1000):
    if viewer.is_alive:
        mujoco.mj_resetData(model, data)
        apply_random_force(data)
        
        for i in range(PRE_SUBSTEPS):
            mujoco.mj_step(model, data)
            viewer.render()
        data.xfrc_applied.fill(0)
        for i in range(POST_SUBSTEPS):
            mujoco.mj_step(model, data)
            viewer.render()
            
        # mujoco.mj_step(model, data, nstep=PRE_SUBSTEPS)
        # data.xfrc_applied.fill(0)
        # mujoco.mj_step(model, data, nstep=POST_SUBSTEPS)
        
        # img = viewer.read_pixels()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
    else:
        break

# close
viewer.close()