import copy
import numpy as np

import rl_utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNOWN_ENV_CONVERGENCE_CRITERIA = [
    'policy_var_diag',
    'env_solved',
    'mean_reward',
    'min_num_total_updates',
]

def deploy_policy_on_env(env, policy, sum_rewards_over_rollout=True, reset_env=True, aggregate_vec_envs=True):

    env_rewards, envs_solved = deploy_policy_on_envs([env], policy, sum_rewards_over_rollout, reset_env, aggregate_vec_envs)
    env_reward = env_rewards[0]
    env_solved = envs_solved[0]
    return env_reward, env_solved


def deploy_policy_on_envs(envs, policy, sum_rewards_over_rollout=True, reset_envs=True, aggregate_vec_envs=True):

    env_rewards = []
    envs_solved = []

    for env in envs:

        # Reset env
        if reset_envs:
            env_obsv = env.reset()
        else:
            env_obsv = env.observation()

        env_context = env.context()

        env_is_done = False
        env_is_solved = False
        num_env_steps = 0
        env_reward = 0.

        if hasattr(env, 'num_envs') and env.num_envs > 1:
            action_envs = np.zeros((env.num_envs, policy.num_params()))
            for e in range(env.num_envs):
                this_context = env_context[e] if env._vectorize_context_seeds else env_context
                action = policy.action_from_state(num_env_steps, env_obsv[e], this_context)
                action_envs[e] = action

            # Just works for one-step envs for now
            obsv_envs, reward_envs, is_done_envs, info_envs = env.step(action_envs)
            assert np.all(is_done_envs)

            envs_are_solved = np.array([info_envs[e]['is_solved'] for e in range(env.num_envs)])

            if aggregate_vec_envs:
                env_reward = np.mean(reward_envs)

                if hasattr(env, 'solved_reward_thresh'):
                    env_is_solved = env_reward >= env.solved_reward_thresh
                else:
                    # find env with reward closest to mean
                    if np.any(env_reward == reward_envs):
                        # perfect match - use result of matching env
                        x_match = np.flatnonzero(env_reward == reward_envs)[0]
                        env_is_solved = envs_are_solved[x_match]
                    else:
                        # find best of the worst envs and decide
                        idx_envs_worse = np.flatnonzero(reward_envs < env_reward)
                        idx_best_worst_env = idx_envs_worse[np.argmax(reward_envs[idx_envs_worse])]
                        if envs_are_solved[idx_best_worst_env]:
                            env_is_solved = True
                        else:
                            # find worst of the best envs
                            idx_envs_better = np.flatnonzero(env_reward < reward_envs)
                            idx_worst_best_env = idx_envs_better[np.argmin(reward_envs[idx_envs_better])]
                            if not envs_are_solved[idx_worst_best_env]:
                                env_is_solved = False
                            else:
                                # not clear what this means
                                raise NotImplementedError

            else:
                env_reward = reward_envs
                env_is_solved = envs_are_solved

        else:
            while not env_is_done and num_env_steps < policy.num_steps:

                # Calculate action
                action = policy.action_from_state(num_env_steps, env_obsv, env_context)
                env_obsv, this_step_reward, env_is_done, info = env.step(action)
                num_env_steps += 1
                if sum_rewards_over_rollout:
                    env_reward += this_step_reward
                else:
                    env_reward = this_step_reward

                env_is_solved = info['is_solved']

        env_rewards.append(env_reward)
        envs_solved.append(env_is_solved)

        # Reset env
        if reset_envs:
            env.reset()

    return env_rewards, envs_solved


def solve_env_using_reps(env,
                         policy,
                         policy_params_mean_init,
                         policy_params_var_init,
                         num_policy_rollouts_before_reps_update,
                         max_reps_param_updates,
                         env_convergence_criteria,
                         reps_hyperparams={},
                         max_num_reps_attempts=1,
                         enable_sum_rewards_over_rollout=True,
                         debug_info=False,
                         verbose=False,
                         ):

    # Input argument handling
    assert isinstance(env_convergence_criteria, dict), \
        "Expected env_convergence_criteria to be a dict, but it is a {}.".format(type(env_convergence_criteria))

    for criterion in env_convergence_criteria.keys():
        assert criterion in KNOWN_ENV_CONVERGENCE_CRITERIA, \
            "env_convergence_criteria type \"{}\" is not recognized.".format(criterion)

    check_policy_var_diag = True if 'policy_var_diag' in env_convergence_criteria else False
    policy_var_diag_thresh = env_convergence_criteria['policy_var_diag'] if check_policy_var_diag else np.inf
    check_env_solved = True if 'env_solved' in env_convergence_criteria else False
    env_solved_thresh = env_convergence_criteria['env_solved'] if check_env_solved else 0.
    check_mean_reward = True if 'mean_reward' in env_convergence_criteria else False
    mean_reward_thresh = env_convergence_criteria['mean_reward'] if check_mean_reward else -np.inf
    check_min_num_total_updates = True if 'min_num_total_updates' in env_convergence_criteria else False
    min_num_total_updates = env_convergence_criteria['min_num_total_updates'] if check_min_num_total_updates else -np.inf

    assert isinstance(max_num_reps_attempts, int) and max_num_reps_attempts > 0, \
        "Expected max_num_reps_attempts to be a positive integer, but it is not."

    # Set up REPS
    reps_hyperparams_to_use = {
        'rel_entropy_bound': 0.5,
        'min_temperature': 0.00001,
    }
    reps_hyperparams_to_use.update(reps_hyperparams)
    reps = rl_utils.Reps(**reps_hyperparams_to_use)

    reps_converged = False

    policy_params_mean = copy.deepcopy(policy_params_mean_init)
    policy_params_var = copy.deepcopy(policy_params_var_init)

    num_reps_attempts = 0
    num_reps_param_total_updates = 0

    num_policy_rollouts_attempts = []
    num_reps_param_updates_attempts = []
    policy_params_mean_attempts = []
    policy_params_var_attempts = []
    mean_reward_obtained_attempts = []

    while not reps_converged and num_reps_attempts < max_num_reps_attempts:

        num_policy_rollouts = 0
        num_reps_param_updates = 0
        policy_params_for_reps = []
        rewards_for_reps = []
        policy_solved_env = []
        policy_params_mean_this_attempt = []
        policy_params_var_this_attempt = []
        mean_reward_obtained_this_attempt = []

        policy_params_mean_init_this_attempt = copy.deepcopy(policy_params_mean)
        policy_params_var_init_this_attempt = copy.deepcopy(policy_params_var)

        if verbose:
            print('REPS attempt {} of {}: '.format(num_reps_attempts + 1, max_num_reps_attempts))
            print('policy_params_mean_init_this_attempt:')
            print(policy_params_mean_init_this_attempt)
            print('policy_params_var_init_this_attempt:')
            print(policy_params_var_init_this_attempt)

        policy_params_mean_this_attempt.append(policy_params_mean_init_this_attempt.tolist())
        if debug_info:
            init_var = policy_params_var_init_this_attempt.tolist()
        else:
            init_var = np.diag(policy_params_var_init_this_attempt).tolist()
        policy_params_var_this_attempt.append(init_var)

        while not reps_converged and num_reps_param_updates < max_reps_param_updates:
            # Reset env
            if hasattr(env, 'num_envs') and env.num_envs > 1:
                # parallel env mode
                observation = env.reset()
                # Sample policy parameters
                policy_params_envs = np.random.multivariate_normal(
                    mean=policy_params_mean,
                    cov=policy_params_var,
                    size=env.num_envs,
                )
                env_context = env.context()

                num_env_steps = 0
                action_envs = np.zeros((env.num_envs, policy.num_params()))
                for e in range(env.num_envs):
                    policy.update_policy(policy_params_envs[e], env_context)
                    action = policy.action_from_state(num_env_steps, observation[e], env_context)
                    action_envs[e] = action

                # Just works for one-step envs for now
                obsv_envs, reward_envs, is_done_envs, info_envs = env.step(action_envs)
                assert np.all(is_done_envs)

                envs_are_solved = np.array([info_envs[e]['is_solved'] for e in range(env.num_envs)])

                policy_params_for_reps.extend(policy_params_envs.tolist())
                rewards_for_reps.extend(reward_envs.tolist())
                policy_solved_env.extend(envs_are_solved.tolist())
                num_policy_rollouts += env.num_envs

            else:
                observation = env.reset()

                # Sample policy parameters
                policy_params = np.random.multivariate_normal(mean=policy_params_mean, \
                                                            cov=policy_params_var)
                policy.update_policy(policy_params, env.context())

                # Rollout policy
                env_is_done = False
                env_is_solved = False
                num_env_steps = 0
                env_reward_for_reps = 0.

                while (not env_is_done) and (num_env_steps < policy.num_steps):

                    # Calculate new action based on policy here
                    action = policy.action_from_state(num_env_steps, observation, env.context())
                    observation, this_step_reward, env_is_done, info = env.step(action)
                    this_step_reward += policy.reward_from_state(num_env_steps, observation, env.context())

                    if enable_sum_rewards_over_rollout:
                        env_reward_for_reps += this_step_reward
                    else:
                        env_reward_for_reps = this_step_reward

                    env_is_solved = info['is_solved']
                    num_env_steps += 1

                # Add to REPS buffers
                policy_params_for_reps.append(policy_params.tolist())
                rewards_for_reps.append(env_reward_for_reps)
                policy_solved_env.append(env_is_solved)
                num_policy_rollouts += 1

            num_policy_rollouts_this_batch = len(policy_params_for_reps)
            if num_policy_rollouts_this_batch >= num_policy_rollouts_before_reps_update:
                num_policy_rollouts_before_reps_update = 10
                n_times_solved = np.sum(policy_solved_env)
                env_solved_frac = n_times_solved/num_policy_rollouts_this_batch

                # Check if we've converged - do not update if so
                policy_var_diag_under_thresh = np.all(np.diag(policy_params_var) <= policy_var_diag_thresh)
                env_solved_over_thresh = env_solved_frac >= env_solved_thresh
                mean_reward_for_policy = np.mean(rewards_for_reps)
                mean_reward_over_thresh = mean_reward_for_policy >= mean_reward_thresh
                min_num_total_updates_over_thresh = num_reps_param_total_updates >= min_num_total_updates

                if verbose:
                    print('Rewards for this policy: {} +- {} (1 stdev, n={})'.format(mean_reward_for_policy, np.std(rewards_for_reps), len(rewards_for_reps)))
                    print('Solved success rate: {} ({}/{})'.format(env_solved_frac, n_times_solved, num_policy_rollouts_this_batch))

                    if check_policy_var_diag and policy_var_diag_under_thresh:
                        print(" -> Policy parameter diagonal variance has converged.")

                    if check_env_solved and env_solved_over_thresh:
                        print(" -> Environment solved fraction has converged.")

                    if check_mean_reward and mean_reward_over_thresh:
                        print(" -> Mean reward for policy has converged.")

                    if check_min_num_total_updates and min_num_total_updates_over_thresh:
                        print(f" -> Total number of parameter updates has converged.")

                    print("")

                mean_reward_obtained_this_attempt.append(mean_reward_for_policy)

                if (policy_var_diag_under_thresh and
                    env_solved_over_thresh and
                    mean_reward_over_thresh and
                    min_num_total_updates_over_thresh):
                    reps_converged = True
                else:
                    # Run the REPS update
                    policy_params_mean, policy_params_var, reps_info = \
                        reps.policy_from_samples_and_rewards(policy_params_for_reps, \
                                                            rewards_for_reps)
                    policy_params_var_diag = np.diag(policy_params_var)

                    # Debug
                    if verbose:
                        env.store_trajectory()
                        print('New policy param mean:')
                        print(policy_params_mean)

                        # print('New policy param var:')
                        # print(policy_params_var)

                        print('New policy param var diag:')
                        print(policy_params_var_diag)
                        print("")

                    policy_params_mean_this_attempt.append(policy_params_mean.tolist())
                    if debug_info:
                        var_to_keep = policy_params_var.tolist()
                    else:
                        var_to_keep = np.diag(policy_params_var).tolist()
                    policy_params_var_this_attempt.append(var_to_keep)

                    # Reset buffers
                    policy_params_for_reps = []
                    rewards_for_reps = []
                    policy_solved_env = []
                    num_reps_param_updates += 1
                    num_reps_param_total_updates += 1

        num_policy_rollouts_attempts.append(num_policy_rollouts)
        num_reps_param_updates_attempts.append(num_reps_param_updates)
        policy_params_mean_attempts.append(policy_params_mean_this_attempt)
        policy_params_var_attempts.append(policy_params_var_this_attempt)
        mean_reward_obtained_attempts.append(mean_reward_obtained_this_attempt)

        num_reps_attempts += 1
        if not reps_converged and num_reps_attempts < max_num_reps_attempts:
            # Seed the next attempt with this result
            # Keep the same mean and divide the covariance
            policy_params_var = policy_params_var_init_this_attempt/2.

    # We keep lists as lists and don't convert to np arrays
    # The dimensions don't always agree depending on how many updates each attempt takes

    if debug_info:
        policy_params_var_diag_attempts = [
            [
                np.diag(var)
                for var in param_var_attempt
            ]
            for param_var_attempt in policy_params_var_attempts
        ]
    else:
        policy_params_var_diag_attempts = policy_params_var_attempts

    solve_env_info = {
        'reps_converged': reps_converged,
        'policy_params_mean': policy_params_mean,
        'policy_params_var': policy_params_var,
        'env_convergence_criteria': env_convergence_criteria,
        'max_num_reps_attempts': max_num_reps_attempts,
        'num_reps_attempts': num_reps_attempts,
        'num_policy_rollouts': num_policy_rollouts_attempts,
        'num_reps_param_updates': num_reps_param_updates_attempts,
        'history': {
            'policy_params_mean': policy_params_mean_attempts,
            'policy_params_var_diag': policy_params_var_diag_attempts,
            'mean_reward': mean_reward_obtained_attempts,
        }
    }

    if debug_info:
        solve_env_info['history']['policy_params_var'] = policy_params_var_attempts


    return reps_converged, policy_params_mean, policy_params_var, solve_env_info
