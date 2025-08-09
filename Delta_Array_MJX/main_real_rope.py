import pickle as pkl
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
import time
import threading
from glob import glob

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
    manager = Manager()
    batched_queue = manager.Queue()
    response_dict = manager.dict()
    child_proc = multiprocessing.Process(
        target=server_process_main,
        args=(child_conn, batched_queue, response_dict, config),
        daemon=True
    )
    child_proc.start()
    return parent_conn, batched_queue, response_dict, child_proc, config, manager

def send_request(lock, pipe_conn, action_code, data=None):
    with lock:
        request = (action_code, data)
        pipe_conn.send(request)
        response = pipe_conn.recv()
        return response

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    # manager = Manager()
    parent_conn, batched_queue, response_dict, child_proc, config, manager = create_server_process()
    lock = manager.Lock()
    
    current_episode = 0
    current_reward = 0
    avg_reward = 0
    
    algo = config['algo']
    from_NN = False
    if algo in ['MATSAC', "MABC", "MABC_Finetune", "MABC_Finetune_Bin", "MABC_Finetune_PB", "MABC_Finetune_CA", "MABC_Finetune_PB_CA"]:
        from_NN = True
        
    obj_name = "rope"
        
    env = delta_array_real_rope.DeltaArrayRealRope(config)
    
    mp4s = glob(f"./data/videos/rope/*{algo}.mp4")
    vid_name = f"video_{len(mp4s)}_{algo}.mp4".strip()

    # lock = threading.Lock()
    stop_event, cap_thread = delta_array_real_rope.start_capture_thread(lock, config['rl_device'], config['traditional'], env.plane_size, save_vid=config['save_vid'], vid_name=vid_name)
    time.sleep(10)
    
    act_grasp = env.soft_reset()
    reward = 0
    try_id = 0
    
    print(f'Algo: {algo} Object: {obj_name}')
    
    algo_dict = {}
    algo_dict[obj_name] = []
    while reward < 20:
        try_id += 1
        push_states = (algo, env.init_state[:env.n_idxs], env.pos[:env.n_idxs], True)
        if algo == "Vis Servo":
            actions = env.vs_action(act_grasp, random=False)
        elif algo == "Random":
            actions = env.vs_action(act_grasp, random=True)
        else:
            actions = send_request(lock, parent_conn, TT_GET_ACTION, push_states)
        
        dist, reward = env.rollout(act_grasp, actions, from_NN)
        print(f"Episode: {current_episode}, Try: {try_id}, Reward: {reward}")
        step_data = {
                'try_id'        : try_id,
                'init_bd_pts'   : env.init_bd_pts,
                'goal_bd_pts'   : env.goal_bd_pts,
                'final_bd_pts'  : env.final_bd_pts,
                'dist'          : dist,
                'reward'        : float(reward),
                'robot_indices' : (env.active_idxs.copy()),
                'actions'       : actions,
                'robot_count'   : (env.n_idxs),
            }
        algo_dict[obj_name].append(step_data)
        
        act_grasp = env.soft_reset()
            
    pkls = glob(f"./data/test_rope/*{algo}.pkl")
    save_path = f"./data/test_rope/test_rope_{len(pkls)}_{algo}.pkl"
    pkl.dump(algo_dict, open(save_path, "wb"))
    
    print("Sending stop signal to capture thread...")
    gc.collect()
    stop_event.set()
    cap_thread.join()
    
    child_proc.join()