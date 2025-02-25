import torch
import wandb
import logging
import numpy as np
import multiprocessing
from threading import Lock
import os

import utils.SAC.sac as sac
import utils.MATSAC.matsac_no_autoreg as matsac
import utils.MABC.madptest as madp_test
import utils.MABC.mabc_test as mabc
import utils.MABC.mabc_finetune as mabc_finetune
import baselines.MATPPO as mat

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
            'algo'              : config['algo'],
            'data_type'         : config['data_type'],
            "dont_log"          : config['dont_log'],
            "rblen"             : config['rblen'],
            'seed'              : 69420,
            "data_dir"          : "./data/rl_data",
            
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
            'act_limit'         : 0.03,

            # Multi Agent Part Below:
            'state_dim'         : 6,
            'action_dim'        : 2,
            "dev_rl"            : config['dev_rl'],
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : config['gradnorm'],
            "delta_array_size"  : [8,8],
            'masked'            : config['masked'],
            'gauss'             : config['gauss'],
            'learned_alpha'     : config['la'],
            'ca'                : config['compa'],
            'test_traj'         : config['test_traj'],
            'test_algos'        : ['MABC', 'Random', 'Vis Servo', 'MATSAC', 'MABC Finetuned'],
            'resume'            : config['resume'] != "No",
            'attn_mech'         : config['attn_mech'],
            'pos_embed'         : config['pos_embed'],
            
            # PPO Params
            'ppo_clip'          : config['ppo_clip'],
            'H_coef'            : config['H_coef'],
            'V_coef'            : config['V_coef'],
            'gae_lambda'        : config['gae_lambda'],
            'nenv'              : config['nenv'],
        }

        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, ma=False, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')

        if self.hp_dict['test_traj']:
            self.pushing_agents = {
                "Random" : None,
                "Vis Servo" : None,
                # "MATSAC" : matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="test"),
                # "MABC" : mabc.MABC(self.hp_dict),
                # "MABC Finetuned" : mabc_finetune.MABC_Finetune(self.hp_dict),
            }
            # self.pushing_agents["MATSAC"].load_saved_policy('./data/rl_data/matsac_mj_final/pyt_save/model.pt')
            # self.pushing_agents["MABC"].load_saved_policy('./utils/MABC/mabc_new_data_ac_gauss.pth')
            # # TODO: Make these 3 things proper
            # self.pushing_agents["MABC Finetuned"].load_saved_policy('./utils/MABC/mabc_new_data_ac_gauss.pth')
        else:
            if config['algo']=="MATSAC":
                self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="train")
                if config['resume'] != "No":
                    self.pushing_agent.load_saved_policy(config['resume'])
            elif config['algo']=="MATPPO":
                self.pushing_agent = mat.MATPPO(ma_env_dict, self.hp_dict, train_or_test="train")
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
                    
    def toggle_pushing_agent(self, algo):
        if algo in self.pushing_agents:
            self.pushing_agent = self.pushing_agents[algo]
        else:
            print(f"Invalid algo: {algo}")

def server_process_main(pipe_conn, config):
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
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    server = DeltaArrayServer(config)

    while True:
        try:
            request = pipe_conn.recv()
            endpoint, data = request
            response = {}

            # Handle "-1" or "exit" request: time to shut down
            if endpoint == -1:
                response = {"status" : True}
                pipe_conn.send(response)
                break

            elif endpoint == MA_GET_ACTION:
                with update_lock:
                    action = server.pushing_agent.get_actions(*data)
                response = action

            elif endpoint == MA_UPDATE_POLICY:
                with update_lock:
                    server.pushing_agent.update(*data)

            elif endpoint == SAVE_MODEL:
                server.pushing_agent.save_policy()

            elif endpoint == LOAD_MODEL:
                server.pushing_agent.load_saved_policy()

            elif endpoint == MARB_STORE:
                server.pushing_agent.ma_replay_buffer.store(data)

            elif endpoint == MARB_SAVE:
                server.pushing_agent.ma_replay_buffer.save_RB()

            elif endpoint == LOG_INFERENCE:
                if not server.hp_dict["dont_log"]:
                    wandb.log({"Inference Reward": data[0], "ep len": data[1], "Global Steps": data[2]})

            elif endpoint == TOGGLE_PUSHING_AGENT:
                server.toggle_pushing_agent(data)

            else:
                print(f"[Child] Invalid endpoint: {endpoint}")

            pipe_conn.send(response)

        except EOFError:
            break

    # Cleanup
    pipe_conn.close()
    print("[Child] Exiting child process safely.")
