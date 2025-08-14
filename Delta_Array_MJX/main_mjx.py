import time
import warnings
import multiprocessing
from multiprocessing import get_context

warnings.filterwarnings("ignore")

class MjxRunner:
    def __init__(self, env, inference_env, config, rb_req_queue, rb_resp_queue, log_queue):
        self.env = env
        self.inference_env = inference_env
        self.config = config
        self.n_envs = config['nenv']
        self.bs = config['bs']
        self.warmup_steps = config['warmup']
        self.n_updates = config['n_updates']
        self.infer_every = config['infer_every']
        self.max_eps = config['explen']
        
        
        self.key = jax.random.PRNGKey(config['seed'])
        
        self.key, agent_key = jax.random.split(self.key)
        self.agent = MATSACAgent(agent_key, config)
        
        self.actor_optimizer = optax.adamw(config['pi_lr'])
        self.critic_optimizer = optax.adamw(config['q_lr'])
        self.alpha_optimizer = optax.adamw(config['pi_lr'])
        
        self.actor_opt_state = self.actor_optimizer.init(eqx.filter(self.agent.actor, eqx.is_array))
        self.critic_opt_state = self.critic_optimizer.init(eqx.filter((self.agent.critic1, self.agent.critic2), eqx.is_array))
        self.alpha_opt_state = self.alpha_optimizer.init(eqx.filter(self.agent.log_alpha, eqx.is_array))

        self.update_fn = create_matsac_update_step(config, self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer)

        self.rb_sample_bool = False
        self.rb_req_queue = rb_req_queue
        self.rb_resp_queue = rb_resp_queue
        self.log_queue = log_queue
        
    def update_scheduler(self, step):
        #  !! TODO: Implement a learning rate scheduler if needed
        pass
    
    def run(self):
        reset_keys = jax.random.split(self.key, self.n_envs+2)
        init_key = reset_keys[0]
        self.key = reset_keys[1]
        
        env_state = State(data=self.env.init_data_mjx, 
                        obs=jnp.zeros((64, 6)),
                        init_bd_pts=jnp.zeros((300, 2)),
                        init_nn_bd_pts=jnp.zeros((64, 2)), 
                        goal_nn_bd_pts=jnp.zeros((64, 2)), 
                        active_robot_mask=jnp.zeros(64, dtype=bool), 
                        obj_idx=jnp.zeros((), dtype=jnp.int32),
                        reward=jnp.zeros(()), 
                        done=jnp.zeros(()), 
                        key=init_key
                    )
        env_state = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * self.n_envs), 
            env_state
        )
        reset_state = State(data=self.inference_env.init_data_mjx, 
                        obs=jnp.zeros((64, 6)),
                        init_bd_pts=jnp.zeros((300, 2)),  
                        init_nn_bd_pts=jnp.zeros((64, 2)), 
                        goal_nn_bd_pts=jnp.zeros((64, 2)), 
                        active_robot_mask=jnp.zeros(64, 
                        dtype=bool), 
                        obj_idx=jnp.zeros((), 
                        dtype=jnp.int32),
                        reward=jnp.zeros(()), 
                        done=jnp.zeros(()), 
                        key=init_key
                    )
        reset_state = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * self.n_envs), 
            reset_state
        )
        
        _, actor_static = eqx.partition(self.agent.actor, eqx.is_array)
        @partial(jax.jit, static_argnames=('actor_static',))
        def get_actions_batched(actor_static, actor_params, obs, key):
            actor_model = eqx.combine(actor_params, actor_static)
            dist = jax.vmap(actor_model)(obs, key=jax.random.split(key, obs.shape[0]))
            actions = dist.sample(key=key)
            return actions
        
        vmapped_state = jax.vmap(self.env.step)
        vmapped_reset = jax.vmap(self.env.reset)
        for ep in range(self.max_eps):
            self.key, reset_key = jax.random.split(self.key)
            # Create a unique key for each parallel environment
            ep_reset_keys = jax.random.split(reset_key, self.n_envs) 
            env_state = vmapped_reset(env_state, ep_reset_keys)
            
            actor_params, _ = eqx.partition(self.agent.actor, eqx.is_array)
            self.key, action_key = jax.random.split(self.key)
            actions = get_actions_batched(actor_static, actor_params, env_state.obs, action_key)

            new_env_state = vmapped_state(env_state, actions)

            rb_data = (env_state.obs, actions, new_env_state.active_robot_mask, 
                       new_env_state.reward, new_env_state.obs, new_env_state.done)
            self.rb_req_queue.put((RB_STORE, rb_data))
            if not self.rb_sample_bool:
                self.rb_sample_bool = True
                self.rb_req_queue.put((RB_SAMPLE, self.bs))
                time.sleep(10)
                
            if ep < self.warmup_steps:
                n_upd = self.n_updates // 10
            else:
                n_upd = self.n_updates
                
            for _ in range(n_upd):
                batch_np = self.rb_resp_queue.get()
                batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
                self.agent, self.actor_opt_state, self.critic_opt_state, self.alpha_opt_state, metrics = \
                    self.update_fn(self.agent, batch, self.actor_opt_state, self.critic_opt_state,
                                    self.alpha_opt_state)
                        
            env_state = new_env_state
            # Every x eps, run inference.
            if ep % self.infer_every == 0:
                self.run_inference(reset_state)
           
    def run_inference(self, reset_state):
        print("\nRunning inference...")
        self.key, reset_key = jax.random.split(self.key)
        reset_state = jax.vmap(self.inference_env.reset)(reset_state, reset_key)
        total_reward = jnp.zeros(self.inference_env.num_envs)

        inference_actor_params, inference_actor_static = eqx.partition(self.agent.actor, eqx.is_array)

        @partial(jax.jit, static_argnames=('actor_static',))
        def get_batched_deterministic_actions(self, actor_static, actor_params, observations, keys):
            actor_model = eqx.combine(actor_params, actor_static)
            dist = jax.vmap(actor_model)(observations, key=keys)
            return dist.loc
        
        for _ in range(self.n_inference_steps):
            self.key, action_key = jax.random.split(self.key)
            action_keys = jax.random.split(action_key, self.inference_env.num_envs)
            actions = get_batched_deterministic_actions(
                inference_actor_static, inference_actor_params, reset_state.obs, action_keys
            )
            
            reset_state = self.inference_env.step(reset_state, actions)
            total_reward += reset_state.reward

        avg_reward = total_reward.mean()
        print(f"Inference finished. Episode reward: {avg_reward:.4f}\n")
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
    inference_env = DeltaArrayEnv(config)
    
    runner = MjxRunner(env, inference_env, config, rb_queue, rb_response, log_queue)
    runner.run()