from multiprocessing import Process, Queue, Manager
import time
import argparse
import mujoco.viewer
import numpy as np
import glfw
from threading import Lock
import gc

from src.delta_array_mj import DeltaArrayMJ


def run_env(env_id, env, sim_len, return_dict):
    state = env.reset()
    # env.set_rope_curve()
    stored_data = []
    for t_step in range(sim_len):
        # if t_step == 1:
        #     env.preprocess_state()
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
    parser.add_argument('-simlen', '--simlen', type=int, default=500, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="disc", help="Object to manipulate in sim")
    parser.add_argument('-detstr', "--detection_string", type=str, default="green block", help="Input to detect obj of interest")
    parser.add_argument('-devrl', '--rl_device', type=int, default=0, help='Device on which to run RL policies')
    parser.add_argument('-devv', '--vis_device', type=int, default=1, help='Device on which to run VLMs')
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    parser.add_argument('-algo', "--algo", type=str, default="MATSAC", help="Name of the algorithm to run")
    parser.add_argument('-t', '--test',  action='store_true', help='True to run tests')
    parser.add_argument("-dontlog", "--dont_log", action="store_true", help="Don't Log Experiment")
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    args = parser.parse_args()
    args = vars(args)

    envs = [DeltaArrayMJ(args) for _ in range(args['nenv'])]
    if args['gui'] and args['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")
    
    # run_env(0, envs[0], args['simlen'], {})
    try:
        # while True:
        if args['gui']:
            ret_dict = {}
            run_env(0, envs[0], args['simlen'], ret_dict)
        else:
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
                print(f"Hakuna")
    except KeyboardInterrupt:
        gc.collect()