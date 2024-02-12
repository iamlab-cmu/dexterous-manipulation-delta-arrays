import numpy as np
import jax
from jax import numpy as jp

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

class MJXScene:
    def __init__(self, cfg):
        self.model = mujoco.MjModel.from_xml_path(cfg['xml_path'])
        self.mjx_model = mjx.device_put(self.model)
        
    