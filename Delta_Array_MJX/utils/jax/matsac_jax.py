import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from utils.jax_utils import polyak_update
from utils.jax.gpt_adaln_core import JaxTransformer as TF

# Standalone JIT-compiled functions for performance optimization

@jax.jit
def compute_q_values(q_vals: jnp.ndarray) -> jnp.ndarray:
    """Compute Q-values by summing and squeezing."""
    return jnp.sum(q_vals.squeeze(-1), axis=-1)

@jax.jit
def compute_target_q(rd: jnp.ndarray, gamma: float, d: jnp.ndarray, 
                     q_next_target: jnp.ndarray, alpha: float, next_log_probs: jnp.ndarray) -> jnp.ndarray:
    return rd # + gamma * (1.0 - d) * (q_next_target - alpha * next_log_probs)

@jax.jit 
def compute_critic_loss(q1_pred: jnp.ndarray, q2_pred: jnp.ndarray, q_target: jnp.ndarray) -> jnp.ndarray:
    """Compute MSE loss for critics."""
    return jnp.mean((q1_pred - q_target)**2) + jnp.mean((q2_pred - q_target)**2)

@jax.jit
def compute_actor_loss(alpha: float, log_probs: jnp.ndarray, q_pred: jnp.ndarray) -> jnp.ndarray:
    """Compute actor loss."""
    return jnp.mean((alpha * log_probs).mean(axis=-1) - q_pred)

@jax.jit  
def compute_alpha_loss(log_alpha: jnp.ndarray, log_probs: jnp.ndarray, target_entropy: float) -> jnp.ndarray:
    """Compute alpha (temperature) loss."""
    log_probs_detached = jax.lax.stop_gradient(log_probs)
    return -jnp.exp(log_alpha) * jnp.mean(log_probs_detached + target_entropy)

@jax.jit
def compute_min_q_values(q1_vals: jnp.ndarray, q2_vals: jnp.ndarray) -> jnp.ndarray:
    """Compute minimum of two Q-value arrays."""
    return jnp.minimum(q1_vals, q2_vals)

class MATSACAgent(eqx.Module):
    actor: eqx.Module
    critic1: eqx.Module
    critic2: eqx.Module

    critic1_target: eqx.Module
    critic2_target: eqx.Module
    log_alpha: jnp.array
    
    def __init__(self, key, config):
        actor_key, c1_key, c2_key = jr.split(key, 3)

        act_out_dim = config['act_dim'] * 2 
        q_out_dim = 1
        self.actor = TF(config, out_dim=act_out_dim, is_critic=False, key=actor_key)
        self.critic1 = TF(config, out_dim=q_out_dim, is_critic=True, key=c1_key)
        self.critic2 = TF(config, out_dim=q_out_dim, is_critic=True, key=c2_key)
        
        self.critic1_target = self.critic1
        self.critic2_target = self.critic2
        
        self.log_alpha = jnp.array(-1.609) # e^log_alpha = 0.2 
      
@eqx.filter_vmap(in_axes=(None, 0, 0))
def _vmapped_actor_sample(actor, s, key):
    model_key, sample_key = jr.split(key)
    dist = actor(s, key=model_key)
    return dist.sample_and_log_prob(key=sample_key)

@eqx.filter_vmap(in_axes=(None, 0, 0, 0))
def _vmapped_critic_eval(critic, s, a, key):
    q_vals = critic(s, a, key=key)
    return compute_q_values(q_vals)

@jax.jit
def _critic_loss_fn(critics_to_grad, static_critics, batch,  next_actions_log_probs,  keys, gamma, alpha):
    q1_model, q2_model = critics_to_grad
    q1_target, q2_target = static_critics
    s, a, rd, s_next, d, pos = batch
    next_actions, next_log_probs = next_actions_log_probs
    q1_key, q2_key, q1t_key, q2t_key = keys

    q1_keys_t = jr.split(q1t_key, s_next.shape[0])
    q2_keys_t = jr.split(q2t_key, s_next.shape[0])
    
    q1_next_target = _vmapped_critic_eval(q1_target, s_next, next_actions, q1_keys_t)
    q2_next_target = _vmapped_critic_eval(q2_target, s_next, next_actions, q2_keys_t)
    q_next_target = compute_min_q_values(q1_next_target, q2_next_target)
    
    q_target = compute_target_q(rd, gamma, d, q_next_target, alpha, next_log_probs)

    q1_keys = jr.split(q1_key, s.shape[0])
    q2_keys = jr.split(q2_key, s.shape[0])
    q1_pred = _vmapped_critic_eval(q1_model, s, a, q1_keys)
    q2_pred = _vmapped_critic_eval(q2_model, s, a, q2_keys)

    loss = compute_critic_loss(q1_pred, q2_pred, q_target)
    return loss

@jax.jit
def _actor_loss_fn(actor_to_grad, static_critics, s, alpha, keys, actor_keys_s):
    actions, log_probs = _vmapped_actor_sample(actor_to_grad, s, actor_keys_s)
    q1_model, q2_model = static_critics
    q1_key, q2_key = keys
    
    q1_keys = jr.split(q1_key, s.shape[0])
    q2_keys = jr.split(q2_key, s.shape[0])
    q1_pred = _vmapped_critic_eval(q1_model, s, actions, q1_keys)
    q2_pred = _vmapped_critic_eval(q2_model, s, actions, q2_keys)
    q_pred = compute_min_q_values(q1_pred, q2_pred)
    
    actor_loss = compute_actor_loss(alpha, log_probs, q_pred)
    return actor_loss, log_probs

@jax.jit
def _alpha_loss_fn(log_alpha_to_grad, log_probs, target_entropy):
    log_probs_detached = jax.lax.stop_gradient(log_probs)
    alpha_loss = -jnp.exp(log_alpha_to_grad) * jnp.mean(log_probs_detached + target_entropy)
    return alpha_loss

def create_matsac_update_step(config, pi_optimizer, q_optimizer, a_optimizer):
    critic_loss_and_grad = eqx.filter_value_and_grad(_critic_loss_fn)
    actor_loss_and_grad = eqx.filter_value_and_grad(_actor_loss_fn, has_aux=True)
    alpha_loss_and_grad = eqx.filter_value_and_grad(_alpha_loss_fn)
    
    target_entropy = -float(config['act_dim'])
    gamma = config['gamma']
    tau = config['tau']
    
    @eqx.filter_jit
    def update_step(agent, batch, pi_state, q_state, a_state, key):
        keys = jr.split(key, 7)
        actor_key_s, actor_key_s_next, q1_key, q2_key, q1t_key, q2t_key, alpha_key = keys
        
        s, _, _, s_next, _, pos = batch
        alpha = jnp.exp(agent.log_alpha)
        
        actor_keys_s = jr.split(actor_key_s, s.shape[0])
        actor_keys_s_next = jr.split(actor_key_s_next, s_next.shape[0])
        
        next_actions, next_log_probs = _vmapped_actor_sample(agent.actor, s_next, actor_keys_s_next)
        
        critic_loss_val, critic_grads = critic_loss_and_grad((agent.critic1, agent.critic2), 
                            (agent.critic1_target, agent.critic2_target), batch, (next_actions, next_log_probs), 
                            (q1_key, q2_key, q1t_key, q2t_key), gamma, alpha
                        )
        critic_updates, new_q_state = q_optimizer.update(critic_grads, q_state, (agent.critic1, agent.critic2))
        new_critic1, new_critic2 = eqx.apply_updates((agent.critic1, agent.critic2), critic_updates)
        
        static_critics = (new_critic1, new_critic2)

        (actor_loss_val, log_probs), actor_grads = actor_loss_and_grad(agent.actor, static_critics, 
                                        s, alpha, (q1_key, q2_key), actor_keys_s)
        actor_updates, new_pi_state = pi_optimizer.update(actor_grads, pi_state, agent.actor)
        new_actor = eqx.apply_updates(agent.actor, actor_updates)
        
        alpha_loss_val, alpha_grads = alpha_loss_and_grad(agent.log_alpha, log_probs, target_entropy)
        alpha_updates, new_a_state = a_optimizer.update(alpha_grads, a_state, agent.log_alpha)
        new_log_alpha = eqx.apply_updates(agent.log_alpha, alpha_updates)
        
        new_agent = eqx.tree_at(
            lambda a: (a.actor, a.critic1, a.critic2, a.log_alpha), 
            agent, 
            (new_actor, new_critic1, new_critic2, new_log_alpha)
        )
        
        new_critic1_target = polyak_update(new_critic1, agent.critic1_target, tau)
        new_critic2_target = polyak_update(new_critic2, agent.critic2_target, tau)
        
        new_agent = eqx.tree_at(
            lambda a: (a.critic1_target, a.critic2_target),
            new_agent,
            (new_critic1_target, new_critic2_target)
        )
        
        metrics = {
            'critic_loss': critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': jnp.exp(new_log_alpha)
        }
        
        return new_agent, new_pi_state, new_q_state, new_a_state, metrics

    return update_step