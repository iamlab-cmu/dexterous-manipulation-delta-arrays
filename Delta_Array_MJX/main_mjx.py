import time
import warnings
import multiprocessing
from multiprocessing import get_context
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from utils.jax.matsac_jax import MATSACAgent, create_matsac_update_step
    from src.delta_array_mjx import State
    
warnings.filterwarnings("ignore")

class TrainState(eqx.Module):
    agent: "MATSACAgent"
    actor_opt_state: "optax.OptState"
    critic_opt_state: "optax.OptState"
    alpha_opt_state: "optax.OptState"
    key: "jnp.ndarray"
    
    # Static attributes for the update function
    update_fn: callable = eqx.static_field()
    
@eqx.filter_jit
def _get_actions_batched(agent, obs, key):
    dist = jax.vmap(agent.actor)(obs, key=jax.random.split(key, obs.shape[0]))
    actions = dist.sample(key=key)
    return actions

@eqx.filter_jit
def _get_deterministic_actions_batched(agent, obs, keys):
    dist = jax.vmap(agent.actor)(obs, key=keys)
    return dist.loc

@eqx.filter_jit
def _perform_updates(train_state: TrainState, n_updates, batch: tuple) -> tuple[TrainState, dict]:
    def _update_body(carry, _):
        ts = carry
        new_key, update_key = jax.random.split(ts.key)
        new_agent, new_aos, new_cos, new_aas, metrics = ts.update_fn(
            ts.agent, batch, ts.actor_opt_state, ts.critic_opt_state, ts.alpha_opt_state, update_key
        )
        ts = eqx.tree_at(lambda x: x.agent, ts, new_agent)
        ts = eqx.tree_at(lambda x: x.actor_opt_state, ts, new_aos)
        ts = eqx.tree_at(lambda x: x.critic_opt_state, ts, new_cos)
        ts = eqx.tree_at(lambda x: x.alpha_opt_state, ts, new_aas)
        ts = eqx.tree_at(lambda x: x.key, ts, new_key)
        return ts, metrics
    final_ts, all_metrics = jax.lax.scan(_update_body, train_state, None, length=n_updates)
    return final_ts, jax.tree_util.tree_map(jnp.mean, all_metrics)
    
@eqx.filter_jit
def _inference_episode(step_fn, agent, act_fn, reset_fn, n_envs, n_inference_steps, key):
    def _scan_step(carry, _):
        env_state, current_key = carry
        action_key, next_key = jax.random.split(current_key)
        
        action_keys = jax.random.split(action_key, n_envs)
        actions = act_fn(agent, env_state.obs, action_keys)
        
        new_env_state = step_fn(env_state, actions)
        return (new_env_state, next_key), new_env_state.reward

    reset_key, episode_key = jax.random.split(key)
    env_state = reset_fn(jax.random.split(reset_key, n_envs))
    _, rewards = jax.lax.scan(_scan_step, (env_state, episode_key), None, length=n_inference_steps)
    return rewards.mean()

class MjxRunner:
    def __init__(self, env, config, rb_req_queue, rb_resp_queue, log_queue):
        print(f"Init", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.env = env
        self.config = config
        self.n_envs = config['nenv']
        self.bs = config['bs']
        self.warmup_steps = config['warmup']
        self.n_updates = config['n_updates']
        self.infer_every = config['infer_every']
        self.max_eps = config['explen']
        self.n_inference_steps = 2
        
        self.rb_req_queue = rb_req_queue
        self.rb_resp_queue = rb_resp_queue
        self.log_queue = log_queue
        
        key = jax.random.PRNGKey(config['seed'])
        key, agent_key = jax.random.split(key)
        
        agent = MATSACAgent(agent_key, config)
        actor_optimizer = optax.adamw(config['pi_lr'])
        critic_optimizer = optax.adamw(config['q_lr'])
        alpha_optimizer = optax.adamw(config['pi_lr'])
        
        self.train_state = TrainState(
            agent=agent,
            actor_opt_state=actor_optimizer.init(eqx.filter(agent.actor, eqx.is_array)),
            critic_opt_state=critic_optimizer.init(eqx.filter((agent.critic1, agent.critic2), eqx.is_array)),
            alpha_opt_state=alpha_optimizer.init(eqx.filter(agent.log_alpha, eqx.is_array)),
            key=key,
            update_fn=create_matsac_update_step(config, actor_optimizer, critic_optimizer, alpha_optimizer)
        )
        
        self.get_actions_batched = _get_actions_batched
        self.act_fn = _get_deterministic_actions_batched
        self.perform_updates = _perform_updates
        self.inference_ep = _inference_episode
        print(f"Init Finish", time.strftime("%Y-%m-%d %H:%M:%S"))
        
    def update_scheduler(self, step):
        #  !! TODO: Implement a learning rate scheduler if needed
        pass
    
    def run(self):
        print("Starting Run...", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        key, reset_key = jax.random.split(self.train_state.key)
        self.train_state = eqx.tree_at(lambda x: x.key, self.train_state, key)
        ep_reset_keys = jax.random.split(reset_key, self.n_envs)
        env_state = self.env.reset(ep_reset_keys)
        
        print(f"Running warmup for {self.warmup_steps} steps...", time.strftime("%Y-%m-%d %H:%M:%S"))
        for _ in range(self.warmup_steps):
            key, action_key = jax.random.split(self.train_state.key)
            self.train_state = eqx.tree_at(lambda x: x.key, self.train_state, key)
            
            actions = self.get_actions_batched(self.train_state.agent, env_state.obs, action_key)
            new_env_state = self.env.step(env_state, actions)
            
            # Store transitions in the replay buffer
            rb_data = (env_state.obs, actions, new_env_state.active_robot_mask, 
                       new_env_state.reward, new_env_state.obs, new_env_state.done)
            self.rb_req_queue.put((RB_STORE, rb_data))
            env_state = new_env_state
        
        print("Warmup complete. Starting training...", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.rb_req_queue.put((RB_SAMPLE, self.bs))
        time.sleep(2)
        for ep in range(self.max_eps):
            batch_np = self.rb_resp_queue.get()
            batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
            
            key, action_key, reset_key = jax.random.split(self.train_state.key, 3)
            self.train_state = eqx.tree_at(lambda x: x.key, self.train_state, key)
            
            actions = self.get_actions_batched(self.train_state.agent, env_state.obs, action_key)
            new_env_state = self.env.step(env_state, actions)

            rb_data = (env_state.obs, actions, new_env_state.active_robot_mask, 
                       new_env_state.reward, new_env_state.obs, new_env_state.done)
            self.rb_req_queue.put((RB_STORE, rb_data))

            ep_reset_keys = jax.random.split(reset_key, self.n_envs)
            env_state_reset = self.env.reset(ep_reset_keys)
            env_state = jax.vmap(
                lambda done, new, reset: jax.lax.cond(jnp.any(done), lambda: reset, lambda: new)
            )(new_env_state.done, new_env_state, env_state_reset)
            
            self.train_state, metrics = self.perform_updates(self.train_state, self.n_updates, batch)
            
            if ep % self.infer_every == 0:
                print(f"\n--- Episode {ep}: Running Inference ---")
                print(metrics)
                avg_rew = self.run_inference()
                log_payload = {
                    'actor_loss': metrics['actor_loss'].item(),
                    'critic_loss': metrics['critic_loss'].item(),
                    'alpha': metrics['alpha'].item(),
                    'training_reward': jnp.mean(new_env_state.reward).item(),
                    'inference_reward': avg_rew,
                    'n_episodes': ep,
                }
                self.log_queue.put(log_payload)
        
    def run_inference(self):
        key, inference_key = jax.random.split(self.train_state.key)
        self.train_state = eqx.tree_at(lambda x: x.key, self.train_state, key)
        avg_reward = self.inference_ep(self.env.step, self.train_state.agent, self.act_fn, self.env.reset, 
                                       self.n_envs, self.n_inference_steps, inference_key)
        
        print(f"Inference finished. Average episode reward: {avg_reward:.4f}\n")
        return avg_reward

###################################################
# Main
###################################################
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from functools import partial
    
    from utils import arg_helper
    from utils.jax.matsac_jax import MATSACAgent, create_matsac_update_step
    from utils.jax.rb_jax import rb_worker_continuous
    from utils.jax.logger_jax import logger_worker
    from src.delta_array_mjx import DeltaArrayEnv, State
    from utils.constants import RB_STORE, RB_SAMPLE
    
        
    config = arg_helper.create_sac_config()
    
    multiprocessing.set_start_method('spawn', force=True)
    ctx = get_context("spawn")
    manager = ctx.Manager()
    rb_queue = manager.Queue(1)
    rb_response = manager.Queue(1000)
    rb_proc = ctx.Process(
        target=rb_worker_continuous,
        args=(config, rb_queue, rb_response),
        daemon=True
    )
    rb_proc.start()
    
    log_queue = manager.Queue(100)
    log_proc = ctx.Process(
        target=logger_worker,
        args=(log_queue, config),
        daemon=True
    )
    log_proc.start()
    
    env = DeltaArrayEnv(config)
    
    runner = MjxRunner(env, config, rb_queue, rb_response, log_queue)
    runner.run()