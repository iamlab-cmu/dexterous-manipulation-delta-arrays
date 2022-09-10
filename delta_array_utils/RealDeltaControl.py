import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

plt.ion()

class DeltaRobotEnv():
    def __init__(self, skill):
        self.robot_positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.block_pose = np.array([0, 0, 0, 0, 0, 0])
        self.target_block_pose = np.array([0, 0, 0, 0, 0, 0])

        """ RL Vars """
        self.return_vars = {"observation": None, "reward": None, "done": None, "info": {"is_solved": False}}
        self.trajectories = []
        self.prev_action = None

    def context(self):
        # Some random function of no significance. But don't remove it!! Needed for REPS
        return None

    def store_trajectory(self):
        # Another random function of no significance. But don't remove it!! Needed for REPS
        self.trajectories.append(self.prev_action)

    