import numpy as np
import matplotlib.pyplot as plt
# import manip_utils.SimDeltaControl as SDC
import delta_array_utils.RealDeltaControl as RDC
from rl_util.identity_policy import IdentityLowLevelPolicy
from rl_util import env_interaction
import pickle
import argparse
import os

def run_reps(skill, sim_or_real):
    n_bfs = 5
    # Initialize the environment
    if sim_or_real == "sim":
        env = SDC.DeltaRobotEnv('./config/env.yaml', skill)
    elif sim_or_real == "real":
        env = RDC.DeltaRobotEnv(skill)
    else:
        print("Wrong sim_or_real argument passed. Exiting...")
        return
        
    # Load the policy
    if skill == "skill1":
        policy = IdentityLowLevelPolicy(2)
        mu = np.array([0, 0])
        sigma = 0.50
    elif skill == "tilt":
        policy = IdentityLowLevelPolicy(n_bfs*2)
        mu = np.zeros(n_bfs*2)
        sigma = 0.7
    elif skill == "lift":
        policy = IdentityLowLevelPolicy(n_bfs*2)
        mu = np.zeros(n_bfs*2)
        sigma = 0.7
    
    max_num_reps_attempts = 5
    max_reps_param_updates = 20

    num_policy_rollouts_before_reps_update = 2 * policy.num_params()
    env_convergence_criteria = {"env_solved": 0.9}

    # Set default Mean and Variance for policy
    policy_params_mean_init = np.zeros(policy.num_params()) + mu
    policy_params_var_init = np.eye(policy.num_params()) * sigma

    reps_converged, low_level_policy_params_mean, \
        low_level_policy_params_var, solve_env_info = \
                env_interaction.solve_env_using_reps(env,
                                    policy,   # this is the pol variable above
                                    policy_params_mean_init,
                                    policy_params_var_init,
                                    num_policy_rollouts_before_reps_update,
                                    max_reps_param_updates,
                                    env_convergence_criteria,
                                    max_num_reps_attempts=max_num_reps_attempts,
                                    debug_info=True,
                                    verbose=True,
                                    )

    reps_policy = {'reps_converged': reps_converged,
                'low_level_policy_params_mean': low_level_policy_params_mean,
                'low_level_policy_params_var': low_level_policy_params_var,
                'solve_env_info': solve_env_info}

    os.makedirs(f'./data/{sim_or_real}/{skill}/')
    num_skills = len(os.listdir(f"./data/{sim_or_real}/{skill}/"))
    with open(f'./data/{sim_or_real}/{skill}/{num_skills+1}.pkl', 'wb') as f:
        pickle.dump(reps_policy, f)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', type=str, default='skill1')
    parser.add_argument('--simorreal', type=str, default='sim')
    args = parser.parse_args()
    run_reps(args.skill, args.simorreal)
