import pickle as pkl
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
import time
import threading

import multiprocessing
from multiprocessing import Process, Manager

from ipc_server import server_process_main
from utils import arg_helper

import src.delta_array_real_rope as delta_array_real_rope

MA_GET_ACTION        = 1
MA_UPDATE_POLICY     = 2
MARB_STORE           = 3
MARB_SAVE            = 4
SAVE_MODEL           = 5
LOAD_MODEL           = 6
LOG_INFERENCE        = 7
TOGGLE_PUSHING_AGENT = 8
TT_GET_ACTION        = 9

OBJ_NAMES = ["block", 'cross', 'diamond', 'hexagon', 'star', 'triangle', 'parallelogram', 'semicircle', "trapezium", 'disc'] #'crescent',

###################################################
# Client <-> Server Communication
###################################################
def start_capture_thread(current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None):
    parent_conn, child_conn = multiprocessing.Pipe()
    capture_thread = multiprocessing.Process(target=delta_array_real_rope.capture_and_convert, args=(child_conn, current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name), daemon=True)
    capture_thread.start()
    return parent_conn, child_conn

def create_server_process():
    config = arg_helper.create_sac_config()
    parent_conn, child_conn = multiprocessing.Pipe()
    child_proc = multiprocessing.Process(
        target=server_process_main,
        args=(child_conn, config),
        daemon=True
    )
    child_proc.start()
    return parent_conn, child_proc, config

def send_request(lock, pipe_conn, action_code, data=None):
    with lock:
        request = (action_code, data)
        pipe_conn.send(request)
        response = pipe_conn.recv()
        return response

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    manager = Manager()
    lock = manager.Lock()
    parent_conn, child_proc, config = create_server_process()
    
    current_episode = 0
    current_reward = 0
    avg_reward = 0
    
    algo = config['algo']
    from_NN = False
    if algo in ['MATSAC', "MABC", "MABC_Finetune"]:
        from_NN = True
        
    obj_name = "rope"
        
    env = delta_array_real_rope.DeltaArrayReal(config)
    
    vid_name = f"video_{algo}_{obj_name}.mp4".strip()

    lock = threading.Lock()
    stop_event, cap_thread = delta_array_real_rope.start_capture_thread(lock, config['rl_device'], config['traditional'], env.plane_size, save_vid=config['save_vid'], vid_name=vid_name)
    time.sleep(5)
    
    env.soft_reset()
    
    try_id = 0
    tries = 1
    
    print(f'Algo: {algo} Object: {obj_name}')
    
    algo_dict = {}
    algo_dict[obj_name] = {}
    while env.rope_reward < 20:
        try_id += 1
        print(f"Episode: {current_episode}, Try: {try_id}")
        push_states = (algo, env.init_state[:env.n_idxs], env.pos[:env.n_idxs], True)
        if algo == "Vis Servo":
            actions = env.vs_action(random=False)
        elif algo == "Random":
            actions = env.vs_action(random=True)
        else:
            actions = send_request(lock, parent_conn, TT_GET_ACTION, push_states)
            env.final_state[:env.n_idxs, 4:6] = actions
        
        env.move_robots(env.active_idxs, actions, delta_array_real_rope.LOW_Z, from_NN)
        env.set_rl_states(actions, final=True, test_traj=True)
        dist, reward = env.compute_reward(actions)
        print(reward)
        
        step_data = {
                'try_id'        : try_id,
                'init_qpos'     : env.init_qpos,
                # 'goal_qpos'     : env.goal_qpos,
                'final_qpos'    : env.final_qpos,
                'dist'          : dist,
                'reward'        : float(reward),
                'robot_indices' : (env.active_idxs.copy()),
                'actions'       : actions.tolist(),
                'robot_count'   : (env.n_idxs),
            }
        
        algo_dict[obj_name][traj_name].append(step_data)
        if (reward > 70) or (tries >= 2):
            tries = 1
            next_pose = current_traj.pop(0)
            next_pose[2] += np.pi/2
            env.soft_reset(goal_2Dpose=next_pose)
        else:
            tries += 1
            env.soft_reset()
            
    save_path = f"./data/test_traj/test_traj_{algo}_{traj_name}_{obj_name}.pkl"
    pkl.dump(algo_dict, open(save_path, "wb"))
    
    print("Sending stop signal to capture thread...")
    stop_event.set()
    cap_thread.join()
    
    child_proc.join()