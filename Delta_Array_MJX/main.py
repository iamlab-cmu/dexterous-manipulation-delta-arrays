import pickle as pkl
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

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

OBJ_NAMES = ["block", 'cross', 'diamond', 'hexagon', 'star', 'triangle', 'parallelogram', 'semicircle', "trapezium", 'disc'] #'crescent',

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
# Environment Runner
###################################################
def run_env(env_id, sim_len, n_runs, return_dict, config, inference, pipe_conn, lock):
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    if config['obj_name'] == "ALL":
        obj_name = OBJ_NAMES[np.random.randint(0, len(OBJ_NAMES))]
    else:
        obj_name = config['obj_name']

    if config['save_vid']:
        recorder = VideoRecorder(output_dir="./data/videos", fps=120)
    else:
        recorder = None
        
    if obj_name != "rope":
        env = delta_array_mj.DeltaArrayRB(config, obj_name)
    else:
        env = delta_array_mj.DeltaArrayRope(config, obj_name)

    run_dict = {}
    for nrun in range(n_runs):
        if env.reset():
            # grasp_states = { 'states': env.init_grasp_state[:env.n_idxs].tolist() }
            # env.actions_grasp[:env.n_idxs] = send_request(grasp_states, SA_GET_ACTION)['actions']
            
            # APPLY GRASP ACTIONS
            env.apply_action(env.actions_grasp[:env.n_idxs])
            env.update_sim(sim_len, recorder)

            push_states = (env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
            # APPLY PUSH ACTIONS
            if (config['vis_servo']) or (np.random.rand() < config['vsd']):
                actions = env.vs_action(random=False)
                env.apply_action(actions)
            else:
                actions = send_request(lock, pipe_conn, MA_GET_ACTION, push_states)
                env.final_state[:env.n_idxs, 4:6] = actions
                env.apply_action(actions)
                
            env.update_sim(sim_len, recorder)
            env.set_rl_states(actions, final=True)
            dist, reward = env.compute_reward(actions)
            if env.gui:
                print(reward)

            run_dict[nrun] = {
                "s0": env.init_state[:env.n_idxs],
                "a": actions,
                "p": env.pos[:env.n_idxs],
                "r": reward,
                "s1": env.final_state[:env.n_idxs],
                "d": True if reward > 0.8 else False,
                "N": env.n_idxs,
            }
        else:
            print("Gandlela Env")
            continue

    if recorder is not None:
        recorder.save_video()
    return_dict[env_id] = run_dict

    env.close()
    del env
    gc.collect()

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
    if config['gui']:
        while True:
            run_env(0, config['simlen'], config['nruns'], {}, config, inference, parent_conn, lock)
    else:
        while True:
            if not config['vis_servo']:
                n_threads, n_runs, n_updates = warmup_scheduler.training_schedule(current_episode)
                if outer_loops%10 == 0:
                    inference = True
                    n_threads = 50
                    nruns = 20
                else:
                    inference = False
            else:
                n_threads, n_runs = config['nenv'], config['nruns']
                    
            return_dict = manager.dict()
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
                        if current_episode % 1000 == 0:
                            send_request(lock, parent_conn, MARB_SAVE, {})
                            if current_episode >= config['rblen']:
                                break
            
            if current_episode >= config['explen']:
                send_request(lock, parent_conn, SAVE_MODEL, {})
                break
            outer_loops += 1
            
        gc.collect()

    send_request(lock, parent_conn, action_code=None, data={"endpoint": "exit"})
    child_proc.join()
    print("[Main] Done.")
