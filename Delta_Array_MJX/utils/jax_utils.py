import jax 
import equinox as eqx
import json
import jax.numpy as jnp

def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)

def polyak_update(online_model: eqx.Module, target_model: eqx.Module, tau: float) -> eqx.Module:
    updated_target_params = jax.tree_util.tree_map(
        lambda online, target: (1 - tau) * target + tau * online,
        eqx.filter(online_model, eqx.is_array),
        eqx.filter(target_model, eqx.is_array) 
    )
    return eqx.combine(updated_target_params, target_model)

def save_agent(filename: str, config: dict, agent: eqx.Module):
    with open(filename, 'wb') as f:
        config_json = json.dumps(config)
        f.write((config_json + '\n').encode())
        eqx.tree_serialise_leaves(f, agent)
    print(f"Agent & config saved to {filename}")

def cosine_with_warm_restarts_and_warmup(init_value: float, cycle_steps: int, alpha: float = 0.0, warmup_steps: int = 500):
    def schedule(count):
        t = count % cycle_steps
        warm = init_value * (alpha + (1 - alpha) * (t / warmup_steps))
        decay_len = cycle_steps - warmup_steps
        t_decay  = t - warmup_steps
        cos_decay = 0.5 * (1 + jnp.cos(jnp.pi * t_decay / decay_len))
        decay = init_value * (alpha + (1 - alpha) * cos_decay)
        return jnp.where(t < warmup_steps, warm, decay)
    return schedule
    
def load_agent(filename: str, key:jax.random.PRNGKey) -> tuple[eqx.Module, dict]:
    with open(filename, 'rb') as f:
        config = json.loads(f.readline().decode())    
        raise NotImplementedError(f"Algorithm {config['algo']} not implemented.")
        
        # agent = eqx.tree_deserialise_leaves(f, agent)
    # print(f"Agent & config loaded from {filename}")
    # return agent, config