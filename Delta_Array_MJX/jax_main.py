from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx

model = mujoco.MjModel.from_xml_path("./config/env.xml")
mjx_model = mjx.device_put(model)

robot_id = 0

@jax.vmap
def batched_step(vel):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.mocap_pos.at[robot_id].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    pos = mjx.step(mjx_model, mjx_data).qpos[0]
    return pos

pos = jax.numpy.array((0.01, 0.01, 1.02))
pos = jax.jit(batched_step)(pos)

plt.plot(pos, pos)
plt.show()

# data = mujoco.MjData(model)
# viewer = mujoco_viewer.MujocoViewer(model, data)
# viewer.cam.lookat = np.array((0.13125, 0.1407285, 1.5))
# viewer.cam.distance = 0.85
# viewer.cam.azimuth = 0
# viewer.cam.elevation = 90

# robot_id = 0
# print(data.mocap_pos)

# for i in range(10000):
#     if i==0:
#         data.mocap_pos[robot_id] = (0.01, 0.01, 1.02)
#     if i%200==0:
#         data.mocap_pos[robot_id] = data.mocap_pos[robot_id]*np.array((-1, -1, 1))
#     if viewer.is_alive:
#         mujoco.mj_step(model, data)
#         viewer.render()
#     else:
#         break

# viewer.close()