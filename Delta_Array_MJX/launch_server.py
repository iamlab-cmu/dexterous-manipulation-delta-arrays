from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import wandb
from pathlib import Path
import argparse
from typing import Any

import utils.SAC.sac as sac
import utils.MATSAC.matsac_no_autoreg as matsac
import utils.MABC.madptest as madp_test
import utils.MABC.mabc_test as mabc
import utils.MABC.mabc_finetune as mabc_finetune
import utils.multi_agent_replay_buffer as MARB

from utils.vision_utils import GroundedSAM
from utils.openai_utils.run_utils import setup_logger_kwargs

class DeltaArrayServer():
    def __init__(self, args, train_or_test="test"):
        self.train_or_test = train_or_test
        self.vision_model = GroundedSAM(obj_detection_model=args.obj_detection_model, 
                                        segmentation_model=args.segmentation_model,
                                        device=torch.device(f"cuda:{args.vis_device}"))
        
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
            "exp_name"          : args.name,
            'algo'              : args.algo,
            'data_type'         : args.data_type,
            "dont_log"          : args.dont_log,
            "replay_size"       : 500001,
            'seed'              : 69420,
            "data_dir"          : "./data/rl_data",
            
            # RL params
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : args.qlr,
            "pi_lr"             : args.pilr,
            'obj_name_enc_dim'  : 9,
            "q_eta_min"         : args.q_etamin,
            "pi_eta_min"        : args.pi_etamin,
            "eta_min"           : args.q_etamin,
            "alpha"             : 0.2,
            'optim'             : args.optim,
            'epsilon'           : 1.0,
            "batch_size"        : args.bs,
            "warmup_epochs"     : args.warmup,
            "infer_every"       : 4000,
            "inference_length"  : 10,
            'act_limit'         : 0.03,

            # Multi Agent Part Below:
            'state_dim'         : 6,
            'action_dim'        : 2,
            "dev_rl"            : torch.device(f"cuda:{args.rl_device}"),
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : args.gradnorm,
            "adaln"             : args.adaln,
            "delta_array_size"  : [8,8],
            'masked'            : not args.unmasked,
            'gauss'             : args.gauss,
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

        if args.algo=="MATSAC":
            self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, logger_kwargs, train_or_test="train")
        elif args.algo=="SAC":
            self.pushing_agent = sac.SAC(simplified_ma_env_dict, self.hp_dict, logger_kwargs, ma=True, train_or_test="train")
        elif args.algo=="MADP":
            self.pushing_agent = madp_test.MADP()
        elif args.algo=="MABC":
            self.pushing_agent = mabc.MABC()
        elif args.algo=="MABC_Finetune":
            self.pushing_agent = mabc_finetune.MABC_Finetune(self.hp_dict)

        if (self.train_or_test=="test") and (args.algo=="MATSAC"):
            # self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/{args.name}_s69420/pyt_save/model.pt')
            self.pushing_agent.load_saved_policy(f'./data/rl_data/{args.name}/pyt_save/model.pt')
        elif (self.train_or_test=="test") and (args.algo in ["MADP", "MABC", "MABC_Finetune"]):
            self.pushing_agent.load_saved_policy(f'./utils/MABC/{args.name}.pth')

class VisionRequest(BaseModel):
    img: Any
    label: Any

class VisionResponse(BaseModel):
    bd_pts: Any
    
class SARequest(BaseModel):
    states: Any

class SAResponse(BaseModel):
    actions: Any
    
class MARBRequest(BaseModel):
    s0: Any
    a: Any
    p: Any
    r: Any
    s1: Any
    d: Any
    N: Any
    
class MAInferenceRequest(BaseModel):
    states: Any
    pos: Any
    det: Any
    
class MAInferenceResponse(BaseModel):
    ma_actions: Any
    
class MATrainRequest(BaseModel):
    batch_size: int
    current_episode: int
    n_updates: int
    
class MATrainResponse(BaseModel):
    losses: Any
    
class MATLoadModel(BaseModel):
    path: str
    
class MATLoadModelResponse(BaseModel):
    status: bool
    
app = FastAPI()
@app.post("/vision/")
def vision_endpoint(request: VisionRequest):
    return VisionResponse(bd_pts=server.vision_model.grounded_obj_segmentation(request.img, request.label))

@app.post("/sac/get_actions")
def sac_endpoint(request: SARequest):
    return SAResponse(actions=server.grasping_agent.get_actions(request.states))

@app.post("/ma/get_actions")
def ma_endpoint(request: MAInferenceRequest):
    return MAInferenceResponse(ma_actions=server.pushing_agent.get_actions(request.states, request.pos, request.det))

@app.post("/marl/update")
def ma_endpoint(request: MATrainRequest):
    return MATrainResponse(losses=server.pushing_agent.update(request.batch_size, request.current_episode, request.n_updates))

@app.post("/marb/store")
def marb_store(request: MARBRequest):
    server.pushing_agent.ma_replay_buffer.store(request.s0, request.a, request.p, request.r, request.s1, request.d, request.N)
    return {"status": "success"}

@app.post("/marb/save")
def marb_store():
    server.pushing_agent.ma_replay_buffer.save_RB()

@app.post("/marl/save_model")
def ma_endpoint(request):
    server.pushing_agent.save_policy()
    
@app.post("/marl/load_model")
def ma_endpoint(request: MATLoadModel):
    return MATLoadModelResponse(status=server.pushing_agent.load_saved_policy(request.path))

@app.post("/wandb/log_reward")
def wandb_log_reward(reward: float):
    wandb.log({"Reward": reward})
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for sim/real test/train")
    parser.add_argument("-r", "--real", action="store_true", help="True for Real Robot Expt")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument("-n", "--name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-data", "--data_type", type=str, default=None, help="Use simplified setup with only 4 fingers?")
    parser.add_argument('-detstr', "--detection_string", type=str, default="green block", help="Input to detect obj of interest")
    parser.add_argument("-dontlog", "--dont_log", action="store_true", help="Don't Log Experiment")
    parser.add_argument('-devrl', '--rl_device', type=int, default=1, help='Device on which to run RL policies')
    parser.add_argument('-devv', '--vis_device', type=int, default=1, help='Device on which to run VLMs')
    parser.add_argument("-bs", "--bs", type=int, default=256, help="Batch Size")
    parser.add_argument("-warmup", "--warmup", type=int, default=5000, help="Exploration Cutoff")
    parser.add_argument("-algo", "--algo", type=str, default="MATSAC", help="RL Algorithm")
    parser.add_argument("-pilr", "--pilr", type=float, default=1e-4, help="% of data to use for visual servoing")
    parser.add_argument("-qlr", "--qlr", type=float, default=1e-4, help="% of data to use for visual servoing")
    parser.add_argument("-adaln", "--adaln", action="store_true", help="Use AdaLN Zero Transformer")
    parser.add_argument("-q_etamin", "--q_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-pi_etamin", "--pi_etamin", type=float, default=1e-5, help="% of data to use for visual servoing")
    parser.add_argument("-gradnorm", "--gradnorm", type=float, default=2.0, help="Grad norm for training")
    parser.add_argument("-unmasked", "--unmasked", action="store_true", help="Unmasked Attention Layers")
    parser.add_argument("-optim", "--optim", type=str, default="adam", help="Optimizer to use adam vs sgd")
    parser.add_argument("-odm", "--obj_detection_model", type=str, default="IDEA-Research/grounding-dino-tiny", help="Obj det model from HF")
    parser.add_argument("-sm", "--segmentation_model", type=str, default="facebook/sam-vit-base", help="Seg model from HF")
    parser.add_argument("-gauss", "--gauss", action="store_true", help="Use Gaussian Final Layers")
    args = parser.parse_args()
    
    train_or_test = "test" if args.test else "train"
    server = DeltaArrayServer(args, train_or_test)
    uvicorn.run(app, host="127.0.0.1", port=8000)
    