from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import wandb
from typing import Any

import utils.SAC.sac as sac
import utils.MATSAC.matsac as matsac
import utils.MADP.madptest as madp_test
import utils.MADP.mabc_test as mabc
import utils.MADP.mabc_finetune as mabc_finetune

from utils.vision_utils import GroundedSAM
from utils.openai_utils.run_utils import setup_logger_kwargs

class DeltaArrayServer():
    def __init__(self, vlm_args, rl_args, device):
        self.vision_model = GroundedSAM(obj_detection_model=vlm_args['obj_detection_model'], 
                                        segmentation_model=vlm_args['segmentation_model'],
                                        device=device)
        
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
            "exp_name"          : rl_args.name,
            'algo'              : rl_args.algo,
            'data_type'         : rl_args.data_type,
            'vis_servo'         : rl_args.vis_servo,
            'test_traj'         : rl_args.test_traj,
            "dont_log"          : rl_args.dont_log,
            "replay_size"       : 500001,
            'seed'              : 69420,
            "data_dir"          : "./data/rl_data",
            
            # RL params
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : rl_args.qlr,
            "pi_lr"             : rl_args.pilr,
            'obj_name_enc_dim'  : 9,
            "q_eta_min"         : rl_args.q_etamin,
            "pi_eta_min"        : rl_args.pi_etamin,
            "eta_min"           : rl_args.q_etamin,
            "alpha"             : 0.2,
            'optim'             : rl_args.optim,
            'epsilon'           : 1.0,
            "batch_size"        : rl_args.bs,
            "warmup_epochs"     : rl_args.warmup,
            "robot_frame"       : rl_args.robot_frame,
            "infer_every"       : 4000,
            "inference_length"  : 10,
            'save_videos'       : rl_args.save_vid,
            'act_limit'         : 0.03,

            # Multi Agent Part Below:
            'state_dim'         : 6,
            'action_dim'        : 2,
            "dev_sim"           : torch.device(f"cuda:{rl_args.dev_sim}"),
            "dev_rl"            : torch.device(f"cuda:{rl_args.dev_rl}"),
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "n_layers_dict"     : {'actor': 12, 'critic': 12},
            "dropout"           : 0,
            "max_grad_norm"     : rl_args.gradnorm,
            "adaln"             : rl_args.adaln,
            "delta_array_size"  : [8,8],
            "add_vs_data"       : rl_args.add_vs_data,
            "vs_ratio"          : rl_args.vs_data,
            'print_summary'     : rl_args.print_summary,
            'masked'            : not rl_args.unmasked,
            'cmu_ri'            : rl_args.cmuri,
        }
        logger_kwargs = {}
        if self.train_or_test=="train":
            if not self.hp_dict["dont_log"]:
                logger_kwargs = setup_logger_kwargs(self.hp_dict['exp_name'], 69420, data_dir=self.hp_dict['data_dir'])
                # writer = SummaryWriter(log_dir=f"./tensorboard/{self.hp_dict['exp_name']}")
                wandb.init(project="MARL_Dexterous_Manipulation",
                        config=self.hp_dict,
                        name = self.hp_dict['exp_name'])

        self.grasping_agent = sac.SAC(single_agent_env_dict, self.hp_dict, logger_kwargs, ma=False, train_or_test="test")
        self.grasping_agent.load_saved_policy('./models/trained_models/SAC_1_agent_stochastic/pyt_save/model.pt')

        if rl_args.algo=="MATSAC":
            self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
        elif rl_args.algo=="SAC":
            self.pushing_agent = sac.SAC(simplified_ma_env_dict, self.hp_dict, logger_kwargs, ma=True, train_or_test="train")
        elif rl_args.algo=="MADP":
            self.pushing_agent = madp_test.MADP()
        elif rl_args.algo=="MABC":
            self.pushing_agent = mabc.MABC()
        elif rl_args.algo=="MABC_Finetune":
            self.pushing_agent = mabc_finetune.MABC_Finetune(self.hp_dict)

        if (self.train_or_test=="test") and (not rl_args.behavior_cloning):
            # self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/{args.name}_s69420/pyt_save/model.pt')
            self.pushing_agent.load_saved_policy(f'./data/rl_data/{rl_args.name}/pyt_save/model.pt')
        elif rl_args.behavior_cloning:
            self.pushing_agent.load_saved_policy(f'./utils/MADP/{rl_args.name}.pth')
            

class VisionRequest(BaseModel):
    perception_input: Any

class VisionResponse(BaseModel):
    perception_output: Any
    
class SARequest(BaseModel):
    states: Any

class SAResponse(BaseModel):
    actions: Any
    
class MAInferenceRequest(BaseModel):
    ma_state: Any
    pos: Any
    obj_name: Any
    
class MAInferenceResponse(BaseModel):
    ma_actions: Any
    
class MAInferenceRequest(BaseModel):
    batch_size: int
    current_episode: int
    n_updates: int
    
app = FastAPI()
server = DeltaArrayServer()

@app.post("/vision/")
async def vision_endpoint(request: VisionRequest):
    return VisionResponse(perception_output=server.vision_model.get_perception_output(request.perception_input))

@app.post("/sac/get_action")
async def sac_endpoint(request: SARequest):
    return SAResponse(actions=server.grasping_agent.get_action(request.states))

@app.post("/ma/get_action")
async def ma_endpoint(request: MAInferenceRequest):
    return MAInferenceResponse(ma_actions=server.pushing_agent.get_action(request.ma_state, request.pos, request.obj_name))

@app.post("/marl/update")
async def ma_endpoint(request: MAInferenceRequest):
    server.pushing_agent.update(request.batch_size, request.current_episode, request.n_updates)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)