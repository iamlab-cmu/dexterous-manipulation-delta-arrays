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


MA_GET_ACTION        = 1
MA_UPDATE_POLICY     = 2
MARB_STORE           = 3
MARB_SAVE            = 4
SAVE_MODEL           = 5
LOAD_MODEL           = 6
LOG_INFERENCE        = 7
TOGGLE_PUSHING_AGENT = 8
TT_GET_ACTION        = 9

OBJ_NAMES = ["block", 'cross', 'diamond', 'disc', 'hexagon', 'star', 'triangle', 'parallelogram', 'semicircle', "trapezium"] #'crescent',

###################################################
# Client <-> Server Communication
###################################################
def start_capture_thread(current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None, n_obj=1):
    parent_conn, child_conn = multiprocessing.Pipe()
    capture_thread = multiprocessing.Process(target=delta_array_real.capture_and_convert, args=(child_conn, current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name, n_obj), daemon=True)
    capture_thread.start()
    return parent_conn, child_conn

def create_server_process():
    config = arg_helper.create_sac_config()
    parent_conn, child_conn = multiprocessing.Pipe()
    manager = Manager()
    # lock2 = manager.Lock()
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
    
    traj_data = pkl.load(open("./data/test_traj/test_trajs_new.pkl", "rb"))
    algo = config['algo']
    from_NN = False
    if algo in ['MATSAC', "MABC", "MABC_Finetune", "MABC_Finetune_Bin", "MABC_Finetune_PB", "MABC_Finetune_CA", "MABC_Finetune_PB_CA"]:
        from_NN = True
        
    obj_name = config['obj_name']
    if obj_name == "ALL":
        raise ValueError("Please specify an object name.")
        # obj_name = OBJ_NAMES[np.random.randint(0, len(OBJ_NAMES))]
        
    # traj_name = np.random.choice(list(traj_data.keys()))
    n_obj = config['n_obj']
    traj_name = config['traj_name']  # Set it to snek or cross or square
    current_traj = traj_data[traj_name].tolist()
    
    if obj_name == "rope":
        import src.delta_array_real_rope as delta_array_real
        env = delta_array_real.DeltaArrayRope(config)
    else:
        import src.delta_array_real as delta_array_real
        env = delta_array_real.DeltaArrayReal(config)
    
    vid_name = f"video_{algo}_{traj_name}_{obj_name}.mp4".strip()

    # lock = threading.Lock()
    stop_event, cap_thread = delta_array_real.start_capture_thread(current_traj, lock, config['rl_device'], config['traditional'], env.plane_size, save_vid=config['save_vid'], vid_name=vid_name, n_obj=n_obj)
    time.sleep(10)
    
    init_pose = current_traj.pop(0)
    # init_pose[2] += np.pi/2
    next_pose = current_traj.pop(0)
    # next_pose[2] += np.pi/2
    act_grasp = env.soft_reset(init_pose, next_pose)
    
    try_id = 0
    tries = 1
    
    print(f'Algo: {algo} Object: {obj_name}, Trajectory: {traj_name}')
    
    algo_dict = {}
    algo_dict[obj_name] = {}
    algo_dict[obj_name][traj_name] = []
    while len(current_traj) > 0:
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
                'init_qpos'     : env.init_qpos,
                # 'goal_qpos'     : env.goal_qpos,
                'final_qpos'    : env.final_qpos,
                'dist'          : dist,
                'reward'        : float(reward),
                'robot_indices' : (env.active_idxs.copy()),
                'actions'       : actions,
                'robot_count'   : (env.n_idxs),
            }
        
        algo_dict[obj_name][traj_name].append(step_data)
        if (reward > 75) or (tries >= 3):
            tries = 1
            next_pose = current_traj.pop(0)
            next_pose[2] += np.pi/2
            act_grasp = env.soft_reset(goal_2Dpose=next_pose)
        else:
            if tries==1:
                next_pose = list((np.array(next_pose) + np.array(init_pose))*2/3)
            tries += 1
            act_grasp = env.soft_reset()
            
    save_path = f"./data/test_traj/test_traj_{algo}_{traj_name}_{obj_name}.pkl"
    pkl.dump(algo_dict, open(save_path, "wb"))
    
    print("Sending stop signal to capture thread...")
    gc.collect()
    stop_event.set()
    cap_thread.join()
    
    child_proc.join()