import pickle as pkl
import numpy as np
import gc
import warnings
import time
import os
warnings.filterwarnings("ignore")
from scipy.spatial.transform import Rotation as R

import multiprocessing
from multiprocessing import Process, Manager

from ipc_server import server_process_main
from utils import arg_helper

import src.delta_array_mj as delta_array_mj

###################################################
# Action / Endpoint Mappings
###################################################
MA_GET_ACTION        = 1
MA_UPDATE_POLICY     = 2
MARB_STORE           = 3
MARB_SAVE            = 4
SAVE_MODEL           = 5
LOAD_MODEL           = 6
LOG_INFERENCE        = 7
TOGGLE_PUSHING_AGENT = 8
TT_GET_ACTION        = 9

OBJ_NAMES = ['star', "block", 'hexagon', 'disc', 'cross', 'diamond', 'triangle', 'parallelogram', 'semicircle', "trapezium"]#'crescent', 

###################################################
# Client <-> Server Communication
###################################################
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
    
###################################################
# Test Trajectory
###################################################
def run_test_traj_rb(env_id, sim_len, experiment_data, algo, VideoRecorder, config, inference, pipe_conn, lock):
    print(algo)
    algo_dict = {}
    if config['save_vid']:
        recorder = VideoRecorder(output_dir="./data/videos", fps=30)
    else:
        recorder = None
        
    print(f"Running Test Trajectories for {algo}")
    for obj_name in OBJ_NAMES:

        print(f"{algo}: Object: {obj_name}")
        algo_dict[obj_name] = {}
        env = delta_array_mj.DeltaArrayRB(config, obj_name)
        with open(f"./data/test_traj/test_trajs_new.pkl", "rb") as f:
            data = pkl.load(f)
                    
        for traj_name, path in data.items():
            
            if recorder:
                recorder.start_recording(filename=f"{algo}_{obj_name}_{traj_name}.mp4")
            algo_dict[obj_name][traj_name] = {}
            
            for run_id in range(1):
                algo_dict[obj_name][traj_name][run_id] = []
                path_cp = path.tolist()
                init_pose = path_cp.pop(0)
                next_pose = path_cp.pop(0)
                env.soft_reset(init=init_pose, goal=next_pose)
                
                try_id = 0
                tries = 1
                
                while len(path_cp) > 0:
                    try_id += 1
                    env.apply_action(env.actions_grasp[:env.n_idxs])
                    env.update_sim(sim_len, recorder)
                    
                    push_states = (algo, env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
                    if algo == "Vis Servo":
                        actions = env.vs_action(random=False)
                    elif algo == "Random":
                        actions = env.vs_action(random=True)
                    else:
                        actions = send_request(lock, pipe_conn, TT_GET_ACTION, push_states)
                        env.final_state[:env.n_idxs, 4:6] = actions
                        
                    env.apply_action(actions)
                    env.update_sim(sim_len, recorder)
                    env.set_rl_states(actions, final=True, test_traj=True)
                    dist, reward = env.compute_reward(actions)
                    print(f"Algo: {algo}, Obj: {obj_name}, Traj: {traj_name}, Run: {run_id}, Attempt: {try_id}, Reward: {reward}")
                    
                    step_data = {
                            'try_id'        : try_id,
                            'init_qpos'     : env.init_qpos,
                            'goal_qpos'     : env.goal_qpos,
                            'final_qpos'    : env.final_qpos,
                            'dist'          : dist,
                            'reward'        : float(reward),
                            'tries'         : tries,
                            'robot_indices' : (env.active_idxs.copy()),
                            'actions'       : actions.tolist(),
                            'robot_count'   : (env.n_idxs),
                        }
                    
                    algo_dict[obj_name][traj_name][run_id].append(step_data)
                    
                    # TODO: Set a proper threshold
                    if (reward > 80) or (tries>3):
                        tries = 1
                        next_pose = path_cp.pop(0)
                        env.soft_reset(goal=next_pose)
                    else:
                        tries+=1
                        env.soft_reset()
                        
                if recorder is not None:
                    recorder.stop_recording()
    
    experiment_data[algo] = algo_dict
        
        
def run_test_traj_rope(env_id, sim_len, experiment_data, algo, VideoRecorder, config, inference, pipe_conn, lock):
    algo_dict = {}
    if config['save_vid']:
        recorder = VideoRecorder(output_dir="./data/videos", fps=30)
    else:
        recorder = None
    # send_request(lock, pipe_conn, TOGGLE_PUSHING_AGENT, algo)
    print(f"Running Test Trajectories for {algo}")
    env = delta_array_mj.DeltaArrayRope(config, obj_name="rope")
        
    for run_id in range(50):
        algo_dict[run_id] = []
        env.reset()
        
        for try_no in range(5):
            env.apply_action(env.actions_grasp[:env.n_idxs])
            env.update_sim(sim_len, recorder)
            
            push_states = (algo, env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
            if algo == "Vis Servo":
                actions = env.vs_action(random=False)
            elif algo == "Random":
                actions = env.vs_action(random=True)
            else:
                actions = send_request(lock, pipe_conn, TT_GET_ACTION, push_states)
                env.final_state[:env.n_idxs, 4:6] = actions
                
            env.apply_action(actions)
            env.update_sim(sim_len, recorder)
            env.set_rl_states(actions, final=True, test_traj=True)
            dist, reward = env.compute_reward_long_horizon()
            # print(dist, reward)
            # print(f"Algo: {algo}, Obj: {obj_name}, Traj: {traj_name}, Run: {run_id}, Attempt: {try_id}, Reward: {reward}")
            
            step_data = {
                    'try_id'        : try_no,
                    'init_qpos'     : env.init_rope_pose,
                    'goal_qpos'     : env.goal_rope_pose,
                    'final_qpos'    : env.final_rope_pose,
                    'reward'        : reward,
                    'dist'          : dist,
                    'robot_indices' : (env.active_idxs.copy()),
                    'actions'       : actions.tolist(),
                    'robot_count'   : (env.n_idxs),
                }   
            
            algo_dict[run_id].append(step_data)
            if (reward > 40) :
                break
            else:
                env.soft_reset()
            
                    
    experiment_data[algo] = algo_dict

###################################################
# Main
###################################################
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    from utils.training_helper import TrainingSchedule
    from utils.video_utils import VideoRecorder
    
    parent_conn, child_proc, config = create_server_process()
    if config['gui'] and config['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")

    manager = Manager()
    lock = manager.Lock()
        
    current_episode = 0
    n_updates = config['nenv']
    avg_reward = 0
    warmup_scheduler = TrainingSchedule(max_nenvs=config['nenv'], max_nruns=config['nruns'], max_warmup_episodes=config['warmup'])
    
    if config['test']:
        send_request(lock, parent_conn, LOAD_MODEL)
        inference = True
    else:
        inference = False

    outer_loops = 0
    if config['test_traj']:
            
        if config['gui']:
            algos = ['Vis Servo']
        else:
            algos = ["Random", "Vis Servo", "MATSAC", "MABC_Finetune", 'MABC']
            # algos = ["MABC Finetuned"]
        
        experiment_data = manager.dict()
        processes = []
        for i in range(len(algos)):
            p = Process(target=run_test_traj_rb if config['obj_name']!="rope" else run_test_traj_rope, args=(i, config['simlen'], experiment_data, algos[i], VideoRecorder, config, True, parent_conn, lock))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
            
        # for algo in algos:
        #     run_test_traj_rb(0, config['simlen'], {}, config, True, parent_conn, lock)
            
        final_expt_data = dict(experiment_data)
        save_path = f"./data/test_traj/test_traj_data_rope.pkl"
        pkl.dump(final_expt_data, open(save_path, "wb"))
        # print(f"Saved Test Trajectory Data at {save_path}")
            
    elif config['gui']:
        while True:
            run_env(0, config['simlen'], config['nruns'], {}, config, inference, parent_conn, lock)
    else:
        while True:
            n_threads, n_runs, n_updates = warmup_scheduler.training_schedule(current_episode)
            return_dict = manager.dict()
            
            if outer_loops%10 == 0:
                inference = True
                n_threads = 50
                nruns = 20
            else:
                inference = False

            processes = []
            for i in range(n_threads):
                p = Process(target=run_env, args=(i, config['simlen'], n_runs, return_dict, config, inference, parent_conn, lock))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            if inference:
                rewards = []
                for env_id, run_dict in return_dict.items():
                    rew = 0
                    for n, (nrun, data) in enumerate(run_dict.items()):
                        rew += data['r']
                    if n > 0:
                        rewards.append(rew/n)
                        print(f"Inference Avg Reward: {rew/n}")
                send_request(lock, parent_conn, LOG_INFERENCE, rewards)
            else:
                avg_reward = 0
                replay_data = []
                
                iters = 0
                for env_id, run_dict in return_dict.items():
                    for nrun, data in run_dict.items():
                        iters += 1
                        avg_reward += data['r']
                        replay_data.append((data['s0'], data['a'], data['p'], data['r'], data['s1'], data['d'], data['N']))
                    
                current_episode += iters
                avg_reward /= iters 
                print(f"Episode: {current_episode}, Avg Reward: {avg_reward}")
                
                if not config['test']:
                    send_request(lock, parent_conn, MARB_STORE, replay_data)
                    
                    if not config['vis_servo']:
                        send_request(lock, parent_conn, MA_UPDATE_POLICY, (current_episode, n_updates, avg_reward))
                        
                    if config['collect_data']:
                        if current_episode >= config['rblen']:
                            send_request(lock, parent_conn, MARB_SAVE, {})
                            break
            
            if current_episode >= config['explen']:
                send_request(lock, parent_conn, SAVE_MODEL, {})
                break
            outer_loops += 1
            
        gc.collect()

    send_request(lock, parent_conn, action_code=None, data={"endpoint": "exit"})
    child_proc.join()
    print("[Main] Done.")
