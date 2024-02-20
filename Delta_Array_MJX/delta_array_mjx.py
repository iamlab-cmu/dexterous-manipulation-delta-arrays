import numpy as np
import sys
import pickle as pkl
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=4)
import wandb

from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
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


class DeltaArrayMJX(PipelineEnv):
    def __init__(self, mjcf_path, cfg, **kwargs):
        model = mujoco.MjModel.from_xml_path("./config/env.xml")
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 8
        model.opt.ls_iterations = 8
        
        sys = mjcf.load_model(model)

        physics_steps_per_control_step = 100
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        rng = jax.random.split(rng, 1)
        print(self.sys)
        qpos = self.sys.qpos
        qvel = self.sys.qvel
        data = self.pipeline_init(qpos, qvel)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
        }
        return State(data, obs, reward, done, metrics)
    
    def pipeline_step(self, pipeline_state: Any, action: jax.Array) -> base.State:
        
        return

    def step(self, state:State, action:jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)