import numpy as np
import argparse
import torch
import torch.nn as nn
import os

def parse_args():    
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC) Arguments')
    
    # Basic Arguments
    parser.add_argument("-n", "--name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-n2", "--finetune_name", type=str, default="HAKUNA", help="Expt Name")
    parser.add_argument("-real", "--real", action="store_true", help="Real World?")
    parser.add_argument('-env', '--env_name', type=str, default="humanoid", help='Name of the environment')
    parser.add_argument('-nenv', '--nenv', type=int, default=1, help='Number of parallel envs')
    parser.add_argument('-nruns', '--nruns', type=int, default=20, help='Number of episodes in each parallel env')
    parser.add_argument("-ie", "--infer_every", type=int, default=10, help="Infer Every n Rollouts")
    parser.add_argument("-t", "--test", action="store_true", help="True for Test")
    parser.add_argument("-wu", "--warmup", type=int, default=100000, help="Exploration Cutoff")
    parser.add_argument('-rb', '--rblen', type=int, default=500000, help='Maximum replay buffer size (default: 10,000)')
    parser.add_argument("-resume", "--resume", type=str, default="No", help="Path to ckpt to resume from (Can be diff esp when resuming expt from copied pt files)")
    parser.add_argument("-wb_resume", "--wb_resume", type=str, default=None, help="WandB resume ID")
    parser.add_argument("-ep_resume", "--ep_resume", type=int, default=0, help="Episode to resume training from")
    parser.add_argument('-nupd', '--n_updates', type=int, default=1, help='Number of updates to run')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-fs', '--frameskip', type=int, default=5, help='Number of steps to run sim after sending ctrl')
    parser.add_argument("-dontlog", "--dont_log", action="store_true", help="Don't Log Experiment")
    parser.add_argument("-el", "--explen", type=int, default=3_000_000, help="Episodes to run RL")
    parser.add_argument("-d2u", "--d2u", type=int, default=1, help="Data to Update Ratio")
    parser.add_argument("-savevid", "--save_vid", action="store_true", help="Save Videos at inference")
    parser.add_argument("-cam", "--use_cam", action="store_true", help="Use camera?")
    parser.add_argument("-pd", "--policy_delay", type=int, default=2, help="Policy Update Delay")
    
    # Delta Array Specific Args
    parser.add_argument("-data", "--data_type", type=str, default=None, help="Use Image or State-based RL")
    parser.add_argument("-test_traj", "--test_traj", action="store_true", help="Test on trajectories")
    parser.add_argument('-simlen', '--simlen', type=int, default=600, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="ALL", help="Object to manipulate in sim")
    parser.add_argument('-traj', "--traj_name", type=str, default="snek", help="Traj to manipulate obj over")
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    parser.add_argument("-v", "--vis_servo", action="store_true", help="True for Visual Servoing")
    parser.add_argument("-vsd", "--vsd", type=float, default=0, help="[0 to 1] ratio of data to use for visual servoing")
    parser.add_argument('-rc', '--rope_chunks', type=int, default=50, help='Number of visual rope chunks')
    parser.add_argument("-compa", "--compa", action="store_true", help="cost for more Actions in reward function")
    parser.add_argument("-pb", "--parsimony_bonus", action="store_true", help="reward for fewer Actions in reward function")
    parser.add_argument("-rs", "--reward_scale", type=float, default=0.01, help="Scale reward function by this value")
    parser.add_argument("-cd", "--collect_data", action="store_true", help="Collect data to be stored in RB")

    parser.add_argument("-n_obj", "--n_obj", type=int, default=1, help="To use multiple objs or not. ")
    parser.add_argument("-r2st", "--r2sdeltat", type=float, default=0, help="Delta shift for real2sim")
    parser.add_argument("-r2sr", "--r2sdeltar", type=float, default=0, help="Delta rotate for real2sim")
    
    parser.add_argument("-nrew", "--new_rew", action="store_true", help="New Gaussian reward function - Smooth gradient")
    parser.add_argument("-trad", "--traditional", action="store_true", help="Traditional Pipeline for Vis Servo")
    parser.add_argument("-lewr", "--long_rew", action="store_true", help="Long Horizon reward function - Smooth gradient")
    
    # PPO Specific Args
    parser.add_argument("-ppo", "--ppo", action="store_true", help="Use PPO instead of SAC")
    parser.add_argument('-mel', '--max_ep_len', type=int, default=20, help='Max # of steps till terminate')
    parser.add_argument('-clip', '--ppo_clip', type=float, default=0.2, help='PPO Epsilon')
    parser.add_argument('-hcoef', '--H_coef', type=float, default=0.01, help='Entropy Coefficient')
    parser.add_argument('-vcoef', '--V_coef', type=float, default=1, help='Value Function Coefficient')
    parser.add_argument('-gae', '--gae_lambda', type=float, default=0.95, help='GAE Lambda')
    
    # Diffusion TD3 Specific Args
    parser.add_argument("-diffr", "--diffrew", action="store_true", help="Use Diffusion 'Reward'")
    parser.add_argument('-k_t', '--k_thresh', type=int, default=20, help='final k steps to train DiffPi')
    parser.add_argument("-w_k", "--w_k", type=float, default=1, help="Use exponentially debuffed k values (0.2739 reference value)")
    parser.add_argument('-exp_n', '--exp_noise', type=float, default=0.0, help='Exploration Noise std; mu=0')
    parser.add_argument("-natc", "--natc", action="store_true", help="Expt 1 with noisy actions to critic")
    parser.add_argument("-entropy", "--entropy", type=float, default=0.0, help="Consider Entropy?")
    
    # Required arguments
    parser.add_argument("-algo", "--algo", type=str, required=True,  help="Choose RL Algorithm: [SAC]")
    parser.add_argument("-arch", "--arch", type=str, default="TF",  help="Choose NN Arch: [MLP, TF]")
    
    # Args for Transformer
    parser.add_argument("-nla", "--n_layers_actor", type=int, default=10, help="Number of Layers for actor GPT")
    parser.add_argument("-nlc", "--n_layers_critic", type=int, default=10, help="Number of Layers for critic GPT")
    parser.add_argument("-nle", "--n_layers_encoder", type=int, default=5, help="Number of Layers for encoder GPT")
    parser.add_argument("-nh", "--num_heads", type=int, default=8, help="Number of Heads for Multihead Attention")
    parser.add_argument("-dff", "--dim_ff", type=int, default=128, help="Feed Forward Dimension")
    parser.add_argument("-adaln", "--adaln", action="store_true", help="Use AdaLN Zero Transformer")
    parser.add_argument("-am", "--attn_mech", type=str, default="AdaLN", help="Choose between SA, CA, AdaLN")
    parser.add_argument("-masked", "--masked", action="store_false", help="Masked Attention Layers True by Default")
    parser.add_argument("-pe", "--pos_embed", type=str, default="SCE", help="Choose between SCE, SPE, RoPE")
    
    # Args for MLP
    parser.add_argument('-hd', '--hidden_dims', type=int, nargs='+', default=[256, 256], help='Hidden layer dimensions (default: [256, 256])')
    
    # Hyperparameters with defaults
    parser.add_argument("-optim", "--optim", type=str, default="adamW", help="Optimizer to use adam, adamW or sgd")
    parser.add_argument("-pilr", "--pi_lr", type=float, default=1e-4, help="% of data to use for visual servoing")
    parser.add_argument("-qlr", "--q_lr", type=float, default=1e-4, help="% of data to use for visual servoing")
    parser.add_argument("-pietamin", "--pi_etamin", type=float, default=1e-6, help="% of data to use for visual servoing")
    parser.add_argument("-qetamin", "--q_etamin", type=float, default=1e-6, help="% of data to use for visual servoing")
    parser.add_argument("-gauss", "--gauss", action="store_true", help="Use Gaussian Final Layers")
    parser.add_argument("-gn", "--gradnorm", type=float, default=2.0, help="Grad norm for training")
    parser.add_argument("-namp", "--not_amp", action="store_false", help="Turn off Automatic Mixed Precision")
    parser.add_argument("-tau", "--tau", type=float, default=0.005, help="EMA Tau")
    parser.add_argument('-gamma', '--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('-alpha', '--alpha', type=float, default=0.2, help='Initial temperature parameter (default: 1.2214). Log of this is 0.2')
    parser.add_argument('-la', '--la', action="store_true", help='Whether to automatically tune alpha (default: False)')
    parser.add_argument('-bs', '--bs', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('-ln', '--layer_norm', action="store_true", help='Whether to use layer normalization (default: False)')
    parser.add_argument('-pn', '--penultimate_norm', action="store_true", help='Whether to use Unit Ball Norm for Critic')
    parser.add_argument('-actvn', '--activation', type=str, default='ReLU', choices=['ReLU', 'SiLU', 'GELU'], help='Activation function (default: ReLU)')
    
    # Add any additional config params here
    parser.add_argument('-mac', '--mac', action="store_true", help='Working on Mac?')
    parser.add_argument('-devrl', '--rl_device', type=int, default=1, help='Device on which to run RL policies')
    parser.add_argument('-seed', '--seed', type=int, default=-1, help='Random seed')
    
    return parser.parse_args()

def get_activation(activation_name: str) -> nn.Module:
    activation_dict = {
        'ReLU': nn.ReLU,
        'SiLU': nn.SiLU,
        'GELU': nn.GELU,
    }
    return activation_dict[activation_name]

def create_sac_config():
    args = parse_args()
    config = vars(args)
    if config['seed'] == -1:
        config['seed'] = np.random.randint(0, 100000)
        
    config['max_agents'] = 64
    config['state_dim'] = 6
    config['act_dim'] = 3
    # config['activation'] = get_activation(config['activation'])
    if config['mac']:
        config['dev_rl'] = torch.device("mps")
    else:
        config['dev_rl'] = torch.device(f"cuda:{config['rl_device']}") if config['rl_device']!=-1 else torch.device("cpu")
    
    if config['test']:
        config['dontlog'] = True
        
    config['path'] = f"./models/{config['algo']}/{config['env_name']}/{config['name']}"
    os.makedirs(config['path'], exist_ok=True)
        
    assert config['algo'] in ['Vis Servo', "MATSAC", "MABC_Finetune", "MABC", "MATPPO", "MADP_Finetune", "MABC_Finetune_Bin", "MABC_Finetune_PB", "MABC_Finetune_CA", "MABC_Finetune_PB_CA"]
    assert config['arch'] in ['MLP', 'TF']
    assert config['attn_mech'] in ['SA', 'CA', 'AdaLN']
    
    if (config['resume'] != "No") and (config['wb_resume'] is None):
        raise ValueError("Need to provide WandB Resume ID")
    
    return config