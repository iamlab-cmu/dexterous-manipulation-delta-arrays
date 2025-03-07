import torch
import wandb
import logging
import numpy as np
import multiprocessing
from threading import Lock
import os

from utils.logger import MetricLogger
import utils.SAC.sac as sac
import utils.MATSAC.matsac_no_autoreg as matsac
# import utils.MABC.madptest as madp_test
import utils.MABC.mabc_test as mabc
import utils.MABC.mabc_finetune as mabc_finetune
import utils.multi_agent_replay_buffer as MARB
import utils.MADPTD3.madptd3 as madp_finetune

update_lock = Lock()

class DeltaArrayServer():
    def __init__(self, config):
        self.train_or_test = "test" if config['test'] else "train"
        os.makedirs(f'./data/rl_data/{config['name']}/pyt_save', exist_ok=True)
            
        single_agent_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'observation_space': {'dim': 4},}
        
        ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                    'pi_obs_space'  : {'dim': 6},
                    'q_obs_space'   : {'dim': 6},
                    "max_agents"    : 64,}
        
        simplified_ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 8},
                    'observation_space'  : {'dim': 24},
                    "max_agents"    : 4}
        
        self.hp_dict = {
            # Env Params
            "exp_name"          : config['name'],
            "diff_exp_name"     : "expt_1",
            'algo'              : config['algo'],
            'data_type'         : config['data_type'],
            "dont_log"          : config['dont_log'],
            "rblen"             : config['rblen'],
            'seed'              : 69420,
            "data_dir"          : "./data/rl_data",
            "real"              : config['real'],
            "infer_every"       : config['infer_every'],
            
            # RL params
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : config['q_lr'],
            "pi_lr"             : config['pi_lr'],
            "q_eta_min"         : config['q_etamin'],
            "pi_eta_min"        : config['pi_etamin'],
            "eta_min"           : config['q_etamin'],
            "alpha"             : config['alpha'],
            'optim'             : config['optim'],
            'epsilon'           : 1.0,
            "batch_size"        : config['bs'],
            "warmup_epochs"     : config['warmup'],
            "policy_delay"      : config['policy_delay'],
            'act_limit'         : 0.03,

            # Multi Agent Part Below:
            'state_dim'         : 6,
            'action_dim'        : 2,
            "dev_rl"            : config['dev_rl'],
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : config['dim_ff'],
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : config['gradnorm'],
            "adaln"             : config['adaln'],
            "delta_array_size"  : [8,8],
            'masked'            : config['masked'],
            'gauss'             : config['gauss'],
            'learned_alpha'     : config['la'],
            'ca'                : config['compa'],
            'test_traj'         : config['test_traj'],
            'test_algos'        : ['MABC', 'Random', 'Vis Servo', 'MATSAC', 'MABC_Finetune'],
            'resume'            : config['resume'] != "No",
            'attn_mech'         : config['attn_mech'],
            'pos_embed'         : config['pos_embed'],
            'rope'              : config['obj_name'] == "rope",
            'idx_embed_loc'     : './utils/MADPTD3/idx_embedding_new.pth',
            'k_thresh'          : config['k_thresh'],
            'natc'              : config['natc'],
            'w_k'               : config['w_k'],
            "exp_noise"         : config['exp_noise'],
            "ppo_clip"          : config['ppo_clip'],
            "entropy"           : config['entropy'],
            "denoising_params"  :{
                'n_diff_steps'      : 100,
                'beta_start'        : 0.0001,
                'beta_end'          : 0.02,
                'beta_schedule'     : 'linear',
                'variance_type'     : 'fixed_small_log',
                'clip_sample'       : True ,
                'prediction_type'   : 'epsilon',
            },
        }
        
        self.logger = MetricLogger(dontlog=self.hp_dict["dont_log"])

        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, ma=False, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')

        if self.hp_dict['test_traj']:
            self.pushing_agents = {
                "Random" : None,
                "Vis Servo" : None,
                "MATSAC" : matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="test"),
                "MABC" : mabc.MABC(self.hp_dict),
                "MABC_Finetune" : mabc_finetune.MABC_Finetune(self.hp_dict),
            }
            # if not self.hp_dict['real']:
            if not self.hp_dict['rope']:
                self.pushing_agents["MATSAC"].load_saved_policy("./models/trained_models/matsac.pt")
                self.pushing_agents["MABC"].load_saved_policy("./models/trained_models/mabc.pt")
                self.pushing_agents["MABC_Finetune"].load_saved_policy("./models/trained_models/mabc_finetuned.pt")
            else:
                self.pushing_agents["MATSAC"].load_saved_policy("./models/trained_models/matsac_rope.pt")
                self.pushing_agents["MABC"].load_saved_policy("./models/trained_models/mabc_rope.pt")
                self.pushing_agents["MABC_Finetune"].load_saved_policy("./models/trained_models/mabc_finetuned_rope.pt")
        else:
            if config['algo']=="MATSAC":
                self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="train")
                if config['resume'] != "No":
                    self.pushing_agent.load_saved_policy(config['resume'])
            elif config['algo']=="SAC":
                self.pushing_agent = sac.SAC(simplified_ma_env_dict, self.hp_dict, ma=True, train_or_test="train")
            elif config['algo']=="MADP":
                self.pushing_agent = madp_test.MADP()
            elif config['algo']=="MABC":
                self.pushing_agent = mabc.MABC()
            elif config['algo']=="MABC_Finetune":
                self.pushing_agent = mabc_finetune.MABC_Finetune(self.hp_dict)
                self.pushing_agent.load_saved_policy(f'./utils/MABC/{config['finetune_name']}.pt')
            elif config['algo']=="MADP_Finetune":
                self.pushing_agent = madp_finetune.MADPTD3(ma_env_dict, self.hp_dict, self.logger)
                if config['finetune_name'] != "HAKUNA":
                    self.pushing_agent.load_saved_policy(f'./utils/MADPTD3/{config['finetune_name']}.pt')

            if (self.train_or_test=="test") and (config['algo']=="MATSAC"):
                # self.pushing_agent.load_saved_policy(f'./data/rl_data/{config['name']}/{config['name']}_s69420/pyt_save/model.pt')
                if config['t_path'] is not None:
                    self.pushing_agent.load_saved_policy(config['t_path'])
                else:
                    self.pushing_agent.load_saved_policy(f'./data/rl_data/{config['name']}/pyt_save/model.pt')
            elif (self.train_or_test=="test") and (config['algo'] in ["MADP", "MABC", "MABC_Finetune"]):
                self.pushing_agent.load_saved_policy(f'./utils/MABC/{config['name']}.pth')
        
            if (self.train_or_test=="train") and (not self.hp_dict["dont_log"]):
                if config['resume'] != "No":
                    wandb.init(project="MARL_Dexterous_Manipulation",
                            config=self.hp_dict,
                            name = self.hp_dict['exp_name'],
                            id=config['wb_resume'],
                            resume=True)
                else:
                    wandb.init(project="MARL_Dexterous_Manipulation",
                            config=self.hp_dict,
                            name = self.hp_dict['exp_name'])
                    
    # def toggle_pushing_agent(self, algo):
        # # model.to('cpu')  # Move model back to CPU
        # # del model  # Delete the model variable
        # # torch.cuda.empty_cache()  # Clear GPU memory
        # # gc.collect()  # Run garbage collection (optional but recommended)
        # if algo in self.pushing_agents:
        #     self.pushing_agent = self.pushing_agents[algo]
        # else:
        #     print(f"Invalid algo: {algo}")

def server_process_main(pipe_conn, batched_queue, response_dict, config):
    """
    Runs in the child process:
        1) Creates the DeltaArrayServer instance
        2) Loops, waiting for requests from the parent process
        3) Executes the request (get_actions, update, etc.) and replies

        MA_GET_ACTION       1 : data_args = [states, pos, deterministic]
        MA_UPDATE_POLICY    2 : data_args = [bs, curr_ep, n_upd, avg_rew]
        MARB_STORE          3 : data_args = [replay_data]
        MARB_SAVE           4 : data_args = None
        SAVE_MODEL          5 : data_args = None
        LOAD_MODEL          6 : data_args = None
        LOG_INFERENCE       7 : data_args = [inference rewards]
        EXIT               -1 : data_args = None
    """
        
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
    
    global_batch_size = 1
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    server = DeltaArrayServer(config)

    while True:
        
        if pipe_conn.poll(0.005):
            try:
                request = pipe_conn.recv()
            except EOFError:
                break
            
            endpoint, data = request
            response = {}

            # Handle "-1" or "exit" request: time to shut down
            if (endpoint == -1) or (endpoint is None):
                pipe_conn.send({"status": True})
                break
            
            elif endpoint == SET_BATCH_SIZE:
                print(f"Set Batch Size: {data}")
                global_batch_size = data
                
            elif endpoint == TT_GET_ACTION:
                algo, data = data[0], data[1:]
                with update_lock:
                    action = server.pushing_agents[algo].get_actions(*data)
                response = action

            elif endpoint == MA_UPDATE_POLICY:
                server.pushing_agent.update(*data)

            elif endpoint == SAVE_MODEL:
                server.pushing_agent.save_model()

            elif endpoint == LOAD_MODEL:
                server.pushing_agent.load_model()

            elif endpoint == MARB_STORE:
                server.pushing_agent.ma_replay_buffer.store(data)

            elif endpoint == MARB_SAVE:
                server.pushing_agent.ma_replay_buffer.save_RB()

            elif endpoint == LOG_INFERENCE:
                if not server.hp_dict["dont_log"]:
                    for reward in data:
                        server.logger.add_data("Inference Reward", reward)
                        # wandb.log({"Inference Reward": reward})

            elif endpoint == TOGGLE_PUSHING_AGENT:
                server.toggle_pushing_agent(data)

            else:
                print(f"[Child] Invalid endpoint: {endpoint}")

            pipe_conn.send(response)
        
        if not batched_queue.empty():
            batched_requests = []
            try:
                # Block to get the first request.
                req = batched_queue.get(timeout=5)
                if req[0] == MA_GET_ACTION:
                    batched_requests.append(req)
            except Exception as e:
                print("Error", e)
                pass
            
            while len(batched_requests) < global_batch_size:
                try:
                    req = batched_queue.get(timeout=5)
                    if req[0] == MA_GET_ACTION:
                        batched_requests.append(req)
                except Exception as e:
                    print("TIMEOUT", e)
                    break
                    
            if len(batched_requests) == global_batch_size:
                # Each request is a tuple: (MA_GET_ACTION, data, response_queue)
                batch_size = len(batched_requests)
                max_agents = max(req[1][0].shape[0] for req in batched_requests)
                obs_dim = batched_requests[0][1][0].shape[1]

                # Create padded arrays.
                batched_obs = np.zeros((batch_size, max_agents, obs_dim), dtype=np.float32)
                batched_pos = np.zeros((batch_size, max_agents, 1), dtype=np.int32)

                # Fill in each batch element.
                for i, (_, data, _) in enumerate(batched_requests):
                    obs_i, pos_i, _ = data
                    n_agents = obs_i.shape[0]
                    batched_obs[i, :n_agents, :] = obs_i
                    pos_i = np.array(pos_i)
                    if pos_i.ndim == 1:
                        pos_i = pos_i.reshape(-1, 1)
                    batched_pos[i, :n_agents, :] = pos_i

                with update_lock:
                    actions, a_ks, log_ps, ents = server.pushing_agent.get_actions(batched_obs, batched_pos)
                    
                # Send each individual action back via its corresponding response Queue.
                for i, (_, _, req_id) in enumerate(batched_requests):
                    response_dict[req_id] = (actions[i], a_ks[i], log_ps[i], ents[i])


    # Cleanup
    pipe_conn.close()
    print("[Child] Exiting child process safely.")
