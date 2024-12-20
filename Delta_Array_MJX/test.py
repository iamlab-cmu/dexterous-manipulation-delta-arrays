from multiprocessing import Process, Queue, Manager
import multiprocessing
import time
import argparse
import mujoco.viewer
import numpy as np
import sys
import gc
import requests

import src.delta_array_mj as delta_array_mj

QUERY_VLM           = 0
SA_GET_ACTION       = 1
MA_GET_ACTION       = 2
MA_UPDATE_POLICY    = 3
MARB_STORE          = 4
MARB_SAVE           = 5
SAVE_MODEL          = 6
LOAD_MODEL          = 7

url_dict = {
    QUERY_VLM: "http://localhost:8000/vision",
    SA_GET_ACTION: "http://localhost:8000/sac/get_actions",
    MA_GET_ACTION: "http://localhost:8000/ma/get_actions",
    MA_UPDATE_POLICY: "http://localhost:8000/marl/update",
    MARB_STORE: "http://localhost:8000/marb/store",
    MARB_SAVE: "http://localhost:8000/marb/save",
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

def run_env(env_id, sim_len, return_dict, args, inference=False):
    # Set a unique random seed for this environment
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    env = delta_array_mj.DeltaArrayMJ(args)

    try:
        run_dict = {}
        for nrun in range(args['nruns']):
            env.reset()
            grasp_states = { 'states': env.init_grasp_state[:env.n_idxs].tolist() }

            env.actions_grasp[:env.n_idxs] = send_request(grasp_states, SA_GET_ACTION)['actions']
            env.init_state[:env.n_idxs, 4:6] = env.actions_grasp[:env.n_idxs]
            for t_step in range(sim_len):
                if t_step == 1:
                    env.apply_action(env.actions_grasp[:env.n_idxs])
                env.update_sim()

            push_states = {
                'states': env.init_state[:env.n_idxs].tolist(),
                'pos': env.pos[:env.n_idxs].tolist(),
                'det': inference
            }

            for t_step in range(sim_len):
                if t_step == 1:
                    if args['vis_servo']:
                        env.vs_action()
                        env.apply_action(env.actions[:env.n_idxs])
                    else:
                        print(push_states)
                        response = send_request(push_states, MA_GET_ACTION)
                        if response is None:
                            print("Smth went wrong")
                            raise ValueError("Error in response")
                        env.actions[:env.n_idxs] = np.array(response['ma_actions'])
                        env.final_state[:env.n_idxs, 4:6] = env.actions[:env.n_idxs]
                        env.apply_action(env.actions[:env.n_idxs])
                env.update_sim()

            if env.rope:
                env.final_state[:env.n_idxs, :2] = env.get_final_rope_pose()
            else:
                env.final_state[:env.n_idxs, :2] = env.get_final_obj_pose()
            reward = env.compute_reward()
            # print(f"Env {env_id}, Run {nrun} Reward: {reward}")

            run_dict[nrun] = {
                "s0": push_states['states'],
                "a": env.actions[:env.n_idxs].tolist(),
                "p": push_states['pos'],
                "r": reward,
                "s1": env.final_state[:env.n_idxs].tolist(),
                "d": True if reward > -1 else False,
                "N": env.n_idxs,
            }
        # Store results in the shared return_dict
        return_dict[env_id] = run_dict
    finally:
        # Clean up resources
        env.close()
        del env
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument('-nenv', '--nenv', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('-nruns', '--nruns', type=int, default=4, help='Number of runs in each parallel env before running off-policy updates')
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
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
    parser.add_argument('-rc', '--rope_chunks', type=int, default=50, help='Number of visual rope chunks')
    parser.add_argument('-dontlog', '--dontlog', action="store_true", help='Set to disable logging')
    args = parser.parse_args()
    args = vars(args)

    if args['gui'] and args['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")
    
    update_dict = {
        'batch_size': 256,
        'current_episode': 0,
        'n_updates': args['nenv']
    }
    outer_loops = 0
    inference = False
    
    now = time.time()
    # TODO: Implement testing script to load a pretrained model and run inference.
    if args['gui']:
        run_env(0, args['simlen'], {}, args, True if args['test'] else False)
    else:
        multiprocessing.set_start_method('spawn', force=True)
        while True:
            manager = Manager()
            return_dict = manager.dict()

            processes = []
            for i in range(args['nenv']):
                p = Process(target=run_env, args=(i, args['simlen'], return_dict, args, inference))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            update_dict['current_episode'] += args['nenv']*args['nruns']
            for env_id, run_dict in return_dict.items():
                avg_reward = 0
                for nrun, data in run_dict.items():
                    avg_reward += data['r']
                    send_request(data, MARB_STORE)
                
                avg_reward /= args['nenv']
                if not args['dontlog']:
                    send_request(avg_reward, "wandb/log_reward")
            
            if update_dict['current_episode'] > 20000:
                print("Data Collection Complete.")
                print(f"Time taken: {time.time() - now}")
                send_request(data, MARB_SAVE)
                break
        # send_request(update_dict, MA_UPDATE_POLICY)
            
    # except Exception as e:
    #     print(e)
    #     send_request({}, SAVE_MODEL)
    #     gc.collect()
    # finally:
    #     sys.exit(1)