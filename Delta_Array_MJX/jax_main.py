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
from brax.envs.base import Env, PipelineEnv, State
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

robot_id = 0

class DeltaArray(PipelineEnv):
    def __init__(self, robot_id, offset, **kwargs):
        model = mujoco.MjModel.from_xml_path("./config/env.xml")
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 6
        model.opt.ls_iterations = 6
        
        sys = mjcf.load_model(model)

        physics_steps_per_control_step = 100
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self.robot_id = robot_id
        self.offset = offset

    def reset(self, rng: jp.ndarray) -> State:
        rng = jax.random.split(rng, 1)
        
        return 