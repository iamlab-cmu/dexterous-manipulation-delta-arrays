import pickle as pkl
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from multiprocessing import Process, Manager, Value

from ipc_server_ppo import server_process_main
from utils import arg_helper

import src.delta_array_mj as delta_array_mj

###################################################
# Action / Endpoint Mappings
###################################################
MA_GET_ACTION       = 1
MA_UPDATE_POLICY    = 2
MARB_STORE          = 3
MARB_SAVE           = 4
SAVE_MODEL          = 5
LOAD_MODEL          = 6
LOG_INFERENCE       = 7
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
def run_env(env_id, sim_len, max_ep_len, max_rb_len, ep_len, global_steps, config, pipe_conn, lock):
    n_updates = config['n_updates']
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)
        
    if config['obj_name'] != "rope":
        obj_name = OBJ_NAMES[env_id % len(OBJ_NAMES)]
        env = delta_array_mj.DeltaArrayRB(config, obj_name)
    else:
        env = delta_array_mj.DeltaArrayRope(config, config['obj_name'])

    print(max_ep_len, max_rb_len, ep_len)
    inference = False
    thread_steps = 0
    while global_steps.value <= max_ep_len:
        steps = 0
        if thread_steps%(max_rb_len*20) == 0:
            inference = True
        final_rewards = []
        while steps < max_rb_len:
            replay_data = []
            env.reset(long_horizon=False)
            
            for iters in range(ep_len):
                steps += 1
                thread_steps += 1
                if steps >= max_rb_len:
                    break
                
                env.apply_action(env.actions_grasp[:env.n_idxs])
                env.update_sim(sim_len)

                push_states = (env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
                actions, logp, v = send_request(lock, pipe_conn, MA_GET_ACTION, push_states)
                
                env.final_state[:env.n_idxs, 4:6] = actions
                env.apply_action(actions)
                
                env.update_sim(sim_len, None)
                env.set_rl_states(actions, final=True)                
                done, reward = env.compute_reward_ppo(actions)

                if not inference:
                    replay_data.append((env.init_state[:env.n_idxs].copy(), actions, logp, v, env.pos[:env.n_idxs].copy(), 
                                            reward, env.final_state[:env.n_idxs].copy(), done, env.n_idxs))    
                    with lock:
                        global_steps.value += 1
                env.soft_reset()
                if done:
                    break
                
            if inference:
                inference = False
                print(f"Inference Reward: {reward}")
                send_request(lock, pipe_conn, LOG_INFERENCE, (reward, iters, global_steps.value))
            else:
                final_rewards.append(reward)
                send_request(lock, pipe_conn, MARB_STORE, (env_id, replay_data))
                
        # # Update and Clear RB
        # print(f"Env {env_id} | Mean Reward: {np.mean(final_rewards)}")
        send_request(lock, pipe_conn, MA_UPDATE_POLICY, (env_id, global_steps.value, np.mean(final_rewards), n_updates, done))
        
    env.close()
    del env
    gc.collect()

###################################################
# Main
###################################################
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    parent_conn, child_proc, config = create_server_process()
    if config['gui'] and config['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")

    manager = Manager()
    lock = manager.Lock()
        
    if config['test']:
        send_request(lock, parent_conn, LOAD_MODEL)
        inference = True
    else:
        inference = False

    if config['gui']:
        while True:
            run_env(0, config['simlen'], config['nruns'], {}, config, inference, parent_conn, lock)
    else:
        n_threads, n_runs = config['nenv'], config['nruns']
                
        return_dict = manager.dict()
        global_steps = Value('i', 0)
        processes = []
        for i in range(n_threads):
            p = Process(target=run_env, args=(i, config['simlen'], config['explen'], config['rblen'], config['max_ep_len'], global_steps, config, parent_conn, lock))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        
        send_request(lock, parent_conn, SAVE_MODEL, {})
        
        gc.collect()

    send_request(lock, parent_conn, action_code=None, data={"endpoint": "exit"})
    child_proc.join()
    print("[Main] Done.")