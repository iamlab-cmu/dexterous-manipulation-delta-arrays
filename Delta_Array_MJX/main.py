from multiprocessing import Process, Queue, Manager
import time
import argparse
import mujoco.viewer
import numpy as np
import glfw
from threading import Lock

from delta_array_mj import DeltaArrayMJ

def run_env(env_id, env, n_steps, return_dict):
    state = env.reset()
    
    stored_data = []
    for _ in range(n_steps):
        env.update_sim()
        stored_data.append(env.data)
    
    return_dict[env_id] = stored_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument('-nenv', '--nenv', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-skip', '--skip', type=int, default=100, help='Number of steps to run sim blind')
    parser.add_argument('-simlen', '--simlen', type=int, default=None, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="disc", help="Object to manipulate in sim")
    args = parser.parse_args()
    args = vars(args)

    envs = [DeltaArrayMJ(args) for _ in range(args['nenv'])]
    
    manager = Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(args['nenv']):
        p = Process(target=run_env, args=(i, envs[i], args['simlen'], return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for env_id, data in return_dict.items():
        print(f"Env {env_id}")
