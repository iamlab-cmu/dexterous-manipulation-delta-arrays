import numpy as np
import matplotlib.pyplot as plt
import argparse
import mujoco
import glfw
import mujoco_viewer

import config.delta_array_generator
from base_env import BaseMJEnv

class DeltaArrayMJ(BaseMJEnv):
    def __init__(self, args):
        super().__init__(args)

    def run_sim(self):
        loop = range(self.args.simlen) if self.args.simlen is not None else iter(int, 1)
        for i in loop:
            self.update_sim()

    

if __name__ == "__main__":
    # mjcf_path = './config/env.xml'
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-skip', '--skip', type=int, default=100, help='Number of steps to run sim blind')
    parser.add_argument('-simlen', '--simlen', type=int, default=None, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="disc", help="Object to manipulate in sim")
    
    args = parser.parse_args()

    delta_array_mj = DeltaArrayMJ(args)
    delta_array_mj.run_sim()