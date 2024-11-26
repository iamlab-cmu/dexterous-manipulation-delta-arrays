from multiprocessing import Process, Queue, Manager
import time
import argparse
import mujoco.viewer
import numpy as np
import sys
from threading import Lock
import gc
import pickle as pkl
import requests

import src.delta_array_mj as delta_array_mj

QUERY_VLM           = 0
SA_GET_ACTION       = 1
MA_GET_ACTION       = 2
MA_UPDATE_POLICY    = 3
MARB_STORE          = 4
SAVE_MODEL          = 5
LOAD_MODEL          = 6

url_dict = {
    QUERY_VLM: "http://localhost:8000/vision",
    SA_GET_ACTION: "http://localhost:8000/sac/get_actions",
    MA_GET_ACTION: "http://localhost:8000/ma/get_actions",
    MA_UPDATE_POLICY: "http://localhost:8000/marl/update",
    MARB_STORE: "http://localhost:8000/marb/store",
    SAVE_MODEL: "http://localhost:8000/marl/save_model",
    LOAD_MODEL: "http://localhost:8000/marl/load_model",
}
def send_request(data, action):
    url = url_dict[action]
    response = requests.post(url, json=data)
    if response.status_code != 200:
        print(f"Error in response: {response.status_code}")
        return None
    return response.json()
    
def run_env(env_id, env, sim_len, return_dict, inference=False):
    env.reset()
    grasp_states = {
        'states': env.init_grasp_state[:env.n_idxs].tolist()
    }
    env.actions_grasp[:env.n_idxs] = send_request(grasp_states, SA_GET_ACTION)['actions']
    env.init_state[:env.n_idxs, 4:6] = env.actions_grasp[:env.n_idxs]
    for t_step in range(sim_len):
        if t_step == 1:
            env.apply_action(env.actions_grasp[:env.n_idxs])
        env.update_sim()
        time.sleep(0.1)
        
    env.init_state[:env.n_idxs, 4:6] = env.actions_grasp
    push_states = {
        'states': env.init_state[:env.n_idxs].tolist(),
        'pos': env.pos[:env.n_idxs].tolist(),
        'det': inference
    }
    env.actions[:env.n_idxs] = send_request(push_states, MA_GET_ACTION)
    for t_step in range(sim_len):
        if t_step == 1:
            env.apply_action(env.actions)
        env.update_sim()
        
    final_state = env.get_final_obj_pose()
    reward = env.compute_reward()
    
    # Return dict tuple has to be in that specific order!
    return_dict[env_id] = (
        push_states['states'], 
        env.actions[:env.n_idxs].tolist(), 
        push_states['pos'], 
        reward, 
        final_state, 
        True, 
        env.n_idxs,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument('-nenv', '--nenv', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-skip', '--skip', type=int, default=100, help='Number of steps to run sim blind')
    parser.add_argument('-simlen', '--simlen', type=int, default=500, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="block", help="Object to manipulate in sim")
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    parser.add_argument("-rf", "--robot_frame", action="store_true", help="Robot Frame Yes or No")
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    parser.add_argument("-avsd", "--add_vs_data", action="store_true", help="True for adding visual servoing data")
    parser.add_argument("-vsd", "--vs_data", type=float, help="[0 to 1] ratio of data to use for visual servoing")
    parser.add_argument("-print", "--print_summary", action="store_true", help="Print Summary and Store in Pickle File")
    parser.add_argument("-savevid", "--save_vid", action="store_true", help="Save Videos at inference")
    parser.add_argument("-XX", "--donothing", action="store_true", help="Do nothing to test sim")
    parser.add_argument("-test_traj", "--test_traj", action="store_true", help="Test on trajectories")
    parser.add_argument("-cmuri", "--cmuri", action="store_true", help="Set to use CMU RI trajectory")
    args = parser.parse_args()
    args = vars(args)

    envs = [delta_array_mj.DeltaArrayMJ(args) for _ in range(args['nenv'])]
    if args['gui'] and args['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")
    
    run_env(0, envs[0], args['simlen'], {})
    # update_dict = {
    #     'batch_size': 256,
    #     'current_episode': 0,
    #     'n_updates': args['nenv']
    # }
    # outer_loops = 0
    # inference = False
    
    # TODO: Implement testing script to load a pretrained model and run inference.
#     try:
#         # while True:
#         #     outer_loops += 1
#         #     if outer_loops%50 == 0:
#         #         inference = True
#         #     else:
#         #         inference = False
#         if args['gui']:
#             ret_dict = {}
#             run_env(0, envs[0], args['simlen'], ret_dict)
#         else:
#             manager = Manager()
#             return_dict = manager.dict()
# z

#             update_dict['current_episode'] += args['nenv']
#             avg_reward = 0
#             for env_id, data in return_dict.items():
#                 avg_reward += data[3]
#                 send_request(data, MARB_STORE)
            
#             avg_reward /= args['nenv']
#             send_request(avg_reward, "wandb/log_reward")
            
#             send_request(update_dict, MA_UPDATE_POLICY)
            
#     except KeyboardInterrupt:
#         "Finishing Up... "
#         send_request({}, SAVE_MODEL)
#         gc.collect()
#     finally:
#         sys.exit(1)