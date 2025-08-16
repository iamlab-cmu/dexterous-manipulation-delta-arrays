import numpy as np
import gc
import warnings
import uuid
import time
import multiprocessing
from multiprocessing import Process, Manager

from ipc_server import server_process_main
from utils import arg_helper
import src.delta_array_mj as delta_array_mj

warnings.filterwarnings("ignore")
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
SET_BATCH_SIZE       = 10

OBJ_NAMES = ["block", "cross", "diamond", "hexagon", "star", "triangle", "parallelogram", "semicircle", "trapezium", "disc"]
# OBJ_NAMES = ["hexagon"]

###################################################
# Client <-> Server Communication
###################################################
def create_server_process():
    config = arg_helper.create_sac_config()
    parent_conn, child_conn = multiprocessing.Pipe()
    manager = Manager()
    # Use the manager to create a shared Queue and a shared response dictionary.
    batched_queue = manager.Queue()
    response_dict = manager.dict()
    child_proc = multiprocessing.Process(
        target=server_process_main,
        args=(child_conn, batched_queue, response_dict, config),
        daemon=True
    )
    child_proc.start()
    return parent_conn, batched_queue, response_dict, child_proc, config, manager

def send_request(pipe_conn, action_code, data=None, lock=None, batched_queue=None, response_dict=None):
    """
    For non-batched endpoints, use the Pipe (with lock).
    For MA_GET_ACTION, put a request (with unique id) in the batched_queue and wait for its response in response_dict.
    """
    if action_code == MA_GET_ACTION:
        req_id = str(uuid.uuid4())
        request = (action_code, data, req_id)
        batched_queue.put(request)
        while req_id not in response_dict:
            time.sleep(0.00001)
        response = response_dict.pop(req_id)
        return response
    else:
        with lock:
            request = (action_code, data)
            pipe_conn.send(request)
            response = pipe_conn.recv()
        return response

def open_loop_rollout(env, sim_len, recorder):
    env.apply_action(env.actions_grasp[:env.n_idxs])
    env.update_sim(sim_len, recorder)
    
###################################################
# Environment Runner
###################################################
def run_env(env_id, sim_len, n_runs, return_dict, config, inference, pipe_conn, batched_queue, response_dict, lock):
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    if config['save_vid']:
        from utils.video_utils import VideoRecorder
        recorder = VideoRecorder(output_dir="./data/videos", fps=120)
    else:
        recorder = None
        
    if config['obj_name'] == "rope":
        env = delta_array_mj.DeltaArrayRope(config, config['obj_name'])
    else:
        obj_name = OBJ_NAMES[np.random.randint(0, len(OBJ_NAMES))]
        env = delta_array_mj.DeltaArrayRB(config, obj_name)

    run_dict = {}
    nrun = 0
    while nrun < n_runs:
        if env.reset():
            open_loop_rollout(env, sim_len, recorder)

            push_states = (env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
            if (config['vis_servo']) or (np.random.rand() < config['vsd']):
                actions = env.vs_action(random=False)
                open_loop_rollout(env, sim_len, recorder)
            else:
                actions, a_ks, log_ps, ents = send_request(pipe_conn, MA_GET_ACTION, push_states,
                                       lock=lock, batched_queue=batched_queue, response_dict=response_dict)
                # ** the following 3 lines are imp cos they are sampled with the max N among all threads at server side
                actions = actions[:env.n_idxs]
                if actions.shape[-1] == 3:
                    active_idxs = np.array(env.active_idxs)
                    inactive_idxs = active_idxs[actions[:, 2] > 0]
                    if len(inactive_idxs) > 0:
                        env.set_z_positions(active_idxs=list(inactive_idxs), low=False)
                    execute_actions = env.clip_actions_to_ws(actions[:, :2])
                else:
                    execute_actions = env.clip_actions_to_ws(actions)
                    
                if a_ks is not None:
                    a_ks = a_ks[:, :env.n_idxs]
                    log_ps = log_ps[:, :env.n_idxs]
                env.final_state[:env.n_idxs, 4:6] = execute_actions
                open_loop_rollout(env, sim_len, recorder)
                
            env.set_rl_states(execute_actions, final=True)
            dist, reward = env.compute_reward(actions, inference)
            if env.gui:
                print(reward)

            run_dict[nrun] = {
                "s0": env.init_state[:env.n_idxs],
                "a": actions.copy(),
                "p": env.pos[:env.n_idxs],
                "r": reward,
                "s1": env.final_state[:env.n_idxs],
                "d": True if reward <0.08 else False,
                "N": env.n_idxs,
                "a_ks": a_ks if a_ks is not None else None,
                "log_ps": log_ps if log_ps is not None else None
            }
            nrun += 1
        else:
            print("Gandlela Env")

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
    
    parent_conn, batched_queue, response_dict, child_proc, config, manager = create_server_process()
    if config['gui'] and config['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")

    manager_lock = manager.Lock()
    
    current_episode = config['ep_resume']
    n_updates = config['nenv']
    infer_every = config['infer_every']
    avg_reward = 0
    warmup_scheduler = TrainingSchedule(max_nenvs=config['nenv'], max_nruns=config['nruns'], max_warmup_episodes=config['warmup'], max_nupdates=config['n_updates'])
    
    if config['test']:
        send_request(parent_conn, LOAD_MODEL, lock=manager_lock)
        inference = True
    else:
        inference = False

    outer_loops = 0
    if config['gui']:
        while True:
            run_env(0, config['simlen'], config['nruns'], {}, config, inference,
                    parent_conn, batched_queue, response_dict, manager_lock)
    else:
        while True:
            if not config['vis_servo']:
                n_threads, n_runs, n_updates = warmup_scheduler.training_schedule(current_episode)
                if outer_loops % infer_every == 0:
                    inference = True
                    n_threads = 40
                    n_runs = 10
                else:
                    inference = False
            else:
                n_threads, n_runs = config['nenv'], config['nruns']
            
            # Set the batch_size for batched action generation
            send_request(parent_conn, SET_BATCH_SIZE, n_threads, lock=manager_lock)
            
            return_dict = manager.dict()
            processes = []
            for i in range(n_threads):
                p = Process(target=run_env, args=(i, config['simlen'], n_runs, return_dict, config, inference,
                                                  parent_conn, batched_queue, response_dict, manager_lock))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            if inference:
                rewards = []
                for env_id, run_dict in return_dict.items():
                    for n, (nrun, data) in enumerate(run_dict.items()):
                        rewards.append(data['r'])
                print(f"Inference Avg Reward: {np.mean(rewards)}")
                send_request(parent_conn, LOG_INFERENCE, rewards, lock=manager_lock)
            else:
                log_reward = []
                replay_data = []
                iters = 0
                for env_id, run_dict in return_dict.items():
                    for nrun, data in run_dict.items():
                        iters += 1
                        log_reward.append(data['r'])
                        replay_data.append((data['s0'], data['a'], data['log_ps'], data['a_ks'], data['p'], data['r'], data['s1'], data['d'], data['N']))
                    
                current_episode += iters
                print(f"Episode: {current_episode}, Avg Reward: {np.mean(log_reward)}")
                
                if not config['test']:
                    send_request(parent_conn, MARB_STORE, replay_data, lock=manager_lock)
                    
                    if not config['vis_servo']:
                        send_request(parent_conn, MA_UPDATE_POLICY, (current_episode, n_updates, log_reward),
                                     lock=manager_lock)
                        
                    if config['collect_data']:
                        if current_episode % 1000 == 0:
                            send_request(parent_conn, MARB_SAVE, {}, lock=manager_lock)
                            if current_episode >= config['rblen']:
                                break
            
            if current_episode >= config['explen']:
                send_request(parent_conn, SAVE_MODEL, {}, lock=manager_lock)
                break
            outer_loops += 1
            
        gc.collect()

    send_request(parent_conn, action_code=None, data={"endpoint": "exit"}, lock=manager_lock)
    child_proc.join()
    print("[Main] Done.")
