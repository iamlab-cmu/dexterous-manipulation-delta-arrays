from multiprocessing import Process, Queue, Manager
import multiprocessing
import time
import argparse
import mujoco.viewer
import numpy as np
import sys
import gc
import requests
import warnings
warnings.filterwarnings("ignore")

import src.delta_array_mj as delta_array_mj

QUERY_VLM           = 0
SA_GET_ACTION       = 1
MA_GET_ACTION       = 2
MA_UPDATE_POLICY    = 3
MARB_STORE          = 4
MARB_SAVE           = 5
SAVE_MODEL          = 6
LOAD_MODEL          = 7
LOG_INFERENCE       = 8

url_dict = {
    QUERY_VLM: "http://localhost:8001/vision",
    SA_GET_ACTION: "http://localhost:8001/sac/get_actions",
    MA_GET_ACTION: "http://localhost:8001/ma/get_actions",
    MA_UPDATE_POLICY: "http://localhost:8001/marl/update",
    MARB_STORE: "http://localhost:8001/marb/store",
    MARB_SAVE: "http://localhost:8001/marb/save",
    SAVE_MODEL: "http://localhost:8001/marl/save_model",
    LOAD_MODEL: "http://localhost:8001/marl/load_model",
    LOG_INFERENCE: "http://localhost:8001/log/inference"
}

OBJ_NAMES = ["block", 'crescent', 'cross', 'diamond', 'hexagon', 'star', 'triangle', 'parallelogram', 'semicircle', "trapezium", 'disc']

def send_request(data, action):
    url = url_dict[action]
    response = requests.post(url, json=data)
    if response.status_code != 200:
        print(f"Error in response: {response.status_code}")
        return None
    return response.json()

def run_env(env_id, sim_len, n_runs, return_dict, args, inference=False):
    # Set a unique random seed for this environment
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    if args['obj_name'] == "ALL":
        obj_name = OBJ_NAMES[np.random.randint(0, len(OBJ_NAMES))]
    else:
        obj_name = args['obj_name']

    env = delta_array_mj.DeltaArrayMJ(args, obj_name)

    try:
        run_dict = {}
        try:
            for nrun in range(n_runs):
                if env.reset():
                    grasp_states = { 'states': env.init_grasp_state[:env.n_idxs].tolist() }

                    # env.actions_grasp[:env.n_idxs] = send_request(grasp_states, SA_GET_ACTION)['actions']
                    env.actions_grasp[:env.n_idxs] = env.init_nn_bd_pts - env.raw_rb_pos
                    env.set_rl_states(grasp=True)
                    
                    # APPLY GRASP ACTIONS
                    env.apply_action(env.actions_grasp[:env.n_idxs])
                    env.update_sim(sim_len)

                    push_states = {
                        'states': env.init_state[:env.n_idxs].tolist(),
                        'pos': env.pos[:env.n_idxs].tolist(),
                        'det': inference
                    }

                    # APPLY PUSH ACTIONS
                    if (args['vis_servo']) or (np.random.rand() < args['vsd']):
                        actions = env.vs_action()
                        env.apply_action(actions)
                    else:
                        response = send_request(push_states, MA_GET_ACTION)
                        if response is None:
                            print("Failed to get actions from server.")
                            return
                        actions = np.array(response['ma_actions'])
                        env.final_state[:env.n_idxs, 4:6] = actions
                        env.apply_action(actions)
                        
                    env.update_sim(sim_len)

                    env.set_rl_states(final=True)
                    reward = env.compute_reward()
                    # print(f"Env {env_id}, Run {nrun} Reward: {reward}")

                    run_dict[nrun] = {
                        "s0": push_states['states'],
                        "a": env.actions[:env.n_idxs].tolist(),
                        "p": push_states['pos'],
                        "r": reward,
                        "s1": env.final_state[:env.n_idxs].tolist(),
                        "d": True if reward > 0.8 else False,
                        "N": env.n_idxs,
                    }
                else:
                    print("Gandlela Env")
                    continue
            # Store results in the shared return_dict
        finally:
            return_dict[env_id] = run_dict
    finally:
        # Clean up resources
        env.close()
        del env
        gc.collect()

if __name__ == "__main__":
    from utils.training_helper import TrainingSchedule
        
    def run_env_gui(sim_len, n_runs, args, inference=True):
        # Set a unique random seed for this environment
        seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        if args['obj_name'] == "rope":
            obj_name = args['obj_name']
            env = delta_array_mj.DeltaArrayMJ(args, obj_name)
            
        if args['save_vid']:
            from utils.video_utils import VideoRecorder
            recorder = VideoRecorder(output_dir="./data/videos", fps=120)
            
        for nrun in range(n_runs):
            if args['obj_name'] == "ALL":
                env = delta_array_mj.DeltaArrayMJ(args, obj_name)
                obj_name = OBJ_NAMES[np.random.randint(0, len(OBJ_NAMES))]

            if env.reset():
                grasp_states = { 'states': env.init_grasp_state[:env.n_idxs].tolist() }

                # env.actions_grasp[:env.n_idxs] = send_request(grasp_states, SA_GET_ACTION)['actions']
                env.actions_grasp[:env.n_idxs] = env.init_nn_bd_pts - env.raw_rb_pos
                env.set_rl_states(grasp=True)
                
                # APPLY GRASP ACTIONS
                env.apply_action(env.actions_grasp[:env.n_idxs])
                if args['save_vid']:
                    env.update_sim_recorder(sim_len, recorder)
                else:
                    env.update_sim(sim_len)

                push_states = {
                    'states': env.init_state[:env.n_idxs].tolist(),
                    'pos': env.pos[:env.n_idxs].tolist(),
                    'det': inference
                }

                # APPLY PUSH ACTIONS
                if (args['vis_servo']) or (np.random.rand() < args['vsd']):
                    actions = env.vs_action()
                    # if np.random.rand() < 0.5:
                    #     actions = np.random.uniform(-0.03, 0.03, size=(env.n_idxs, 2))
                    
                    env.apply_action(actions)
                else:
                    response = send_request(push_states, MA_GET_ACTION)
                    if response is None:
                        print("Failed to get actions from server.")
                        return
                    actions = np.array(response['ma_actions'])
                    env.final_state[:env.n_idxs, 4:6] = actions
                    env.apply_action(actions)
                    
                if args['save_vid']:
                    env.update_sim_recorder(sim_len, recorder)
                else:
                    env.update_sim(sim_len)

                env.set_rl_states(final=True)
                reward = env.compute_reward()
                print(f"Obj Name: {obj_name}, Reward: {reward}")
                if args['save_vid']:
                    recorder.save_video()
    
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument('-nenv', '--nenv', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('-nruns', '--nruns', type=int, default=4, help='Number of runs in each parallel env before running off-policy updates')
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-simlen', '--simlen', type=int, default=2000, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="ALL", help="Object to manipulate in sim")
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    parser.add_argument("-vsd", "--vsd", type=float, default=0, help="[0 to 1] ratio of data to use for visual servoing")
    parser.add_argument("-savevid", "--save_vid", action="store_true", help="Save Videos at inference")
    parser.add_argument("-XX", "--donothing", action="store_true", help="Do nothing to test sim")
    parser.add_argument("-test_traj", "--test_traj", action="store_true", help="Test on trajectories")
    parser.add_argument("-cmuri", "--cmuri", action="store_true", help="Set to use CMU RI trajectory")
    parser.add_argument('-rc', '--rope_chunks', type=int, default=50, help='Number of visual rope chunks')
    parser.add_argument("-ca", "--ca", action="store_true", help="cost for Actions in reward function")
    parser.add_argument('-wu', '--warmup', type=int, default=100000, help='Max warmup episodes')
    parser.add_argument("-cd", "--collect_data", action="store_true", help="Collect data to be stored in RB")
    parser.add_argument("-rblen", "--rblen", type=int, default=500000, help="How much data to be stored in RB")
    parser.add_argument("-rs", "--reward_scale", type=float, default=1, help="Scale reward function by this value")
    parser.add_argument("-el", "--explen", type=int, default=3_000_000, help="How long to run RL")
    args = parser.parse_args()
    args = vars(args)

    if args['gui'] and args['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")
    
    update_dict = {
        'batch_size': 256,
        'current_episode': 0,
        'n_updates': args['nenv'],
        'avg_reward': 0
    }
    outer_loops = 0
    inference = False
    
    # TODO: Implement testing script to load a pretrained model and run inference.
    
    warmup_scheduler = TrainingSchedule(max_nenvs=args['nenv'], max_nruns=args['nruns'], max_warmup_episodes=args['warmup'])
    if args['gui'] or (args['nenv']==1):
        while True:
            run_env_gui(args['simlen'], args['nruns'], args)
    else:
        multiprocessing.set_start_method('spawn', force=True)
        while True:
            n_threads, n_runs, update_dict['n_updates'] = warmup_scheduler.training_schedule(update_dict['current_episode'])
            # n_threads, n_runs, update_dict['n_updates'] = args['nenv'], args['nruns'], args['nenv']
            manager = Manager()
            return_dict = manager.dict()

            if outer_loops%10 == 0:
                inference = True
                n_threads = 50
                nruns = 20
            else:
                inference = False
                
            processes = []
            for i in range(n_threads):
                p = Process(target=run_env, args=(i, args['simlen'], n_runs, return_dict, args, inference))
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
                send_request({'rewards': rewards}, LOG_INFERENCE)
            else:
                update_dict['avg_reward'] = 0
                replay_data = {'replay_data': []}
                
                iters = 0
                for env_id, run_dict in return_dict.items():
                    for nrun, data in run_dict.items():
                        iters += 1
                        update_dict['avg_reward'] += data['r']
                        replay_data['replay_data'].append((data['s0'], data['a'], data['p'], data['r'], data['s1'], data['d'], data['N']))
                    
                update_dict['current_episode'] += iters
                update_dict['avg_reward'] /= iters 
                print(f"Episode: {update_dict['current_episode']}, Avg Reward: {update_dict['avg_reward']}")

                
                if not args['test']:
                    send_request(replay_data, MARB_STORE)
                    
                    if args['collect_data']:
                        if update_dict['current_episode'] >= args['rblen']:
                            send_request({}, MARB_SAVE)
                            break
                    if not args['vis_servo']:
                        send_request(update_dict, MA_UPDATE_POLICY)
            
            if update_dict['current_episode'] >= args['explen']:
                send_request({}, SAVE_MODEL)
                break
            outer_loops += 1
            
        gc.collect()