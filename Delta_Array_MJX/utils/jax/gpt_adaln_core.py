import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
from jax.scipy.stats import norm

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6

class TanhNormal(eqx.Module):
    loc: Float[Array, "..."]
    scale: Float[Array, "..."]
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    @classmethod
    def from_loc_and_log_std(cls, loc, log_std):
        scale = jnp.exp(jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return cls(loc, scale)

    def sample(self, *, key: PRNGKeyArray) -> Array:
        noise = jr.normal(key, self.loc.shape)
        unsquashed_action = self.loc + self.scale * noise
        return jnp.tanh(unsquashed_action)

    def log_prob(self, value: Array) -> Array:
        value = jnp.clip(value, -1.0 + EPSILON, 1.0 - EPSILON)
        pre_tanh_value = jnp.arctanh(value)
        log_prob_gaussian = norm.logpdf(pre_tanh_value, loc=self.loc, scale=self.scale).sum(axis=-1)
        log_prob_correction = jnp.log(1.0 - value**2 + EPSILON).sum(axis=-1)
        return log_prob_gaussian - log_prob_correction

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Array, Array]:
        noise = jr.normal(key, self.loc.shape)
        unsquashed_action = self.loc + self.scale * noise
        log_prob_gaussian = norm.logpdf(unsquashed_action, loc=self.loc, scale=self.scale).sum(axis=-1)
        action = jnp.tanh(unsquashed_action)
        log_prob_correction = jnp.log(1.0 - action**2 + EPSILON).sum(axis=-1)
        log_prob = log_prob_gaussian - log_prob_correction
        return action, log_prob
    
    def deterministic_sample(self) -> Array:
        return jnp.tanh(self.loc)

class SCEmbedding(eqx.Module):
    embedding: nn.Embedding
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, key=key1)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim, key=key2)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim, key=key3)

    def __call__(self, x):
        x = self.embedding(x)
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return x

def modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1 + scale) + shift

def orthogonal_init(weight: Array, key: PRNGKeyArray, gain: float = 1.0) -> Array:
    if weight.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    
    rows, cols = weight.shape[-2:]
    flattened_shape = (cols, rows * jnp.prod(jnp.array(weight.shape[:-2])))
    random_matrix = jr.normal(key, flattened_shape)
    q, r = jnp.linalg.qr(random_matrix)
    
    d = jnp.diag(r)
    q *= jnp.sign(d)
    q = q.T * gain
    return q.reshape(weight.shape)

def linear_init(weight: Array, key: PRNGKeyArray) -> Array:
    return orthogonal_init(weight, key, gain=1.0)

def constant_init(bias: Array, value: float = 0.0) -> Array:
    return jnp.full_like(bias, value)

class AdaLMLP(eqx.Module):
    fc1: nn.Linear
    fc2: nn.Linear
    
    def __init__(self, model_dim: int, dim_ff: int, key=None):
        key1, key2 = jax.random.split(key)
        self.fc1 = nn.Linear(model_dim, dim_ff, key=key1)
        self.fc2 = nn.Linear(dim_ff, model_dim, key=key2)

    def __call__(self, x):
        return self.fc2(jax.nn.silu(self.fc1(x)))

class AdaLNAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    split_head_dim: int = eqx.field(static=True)
    masked: bool = eqx.field(static=True)
    chunk_len: int = eqx.field(static=True)

    W_Q: nn.Linear
    W_K: nn.Linear
    W_V: nn.Linear
    W_O: nn.Linear
    tril: jax.Array

    def __init__(self, model_dim: int, num_heads: int, chunk_len: int, masked: bool, key=None):
        keys = jax.random.split(key, 4)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.split_head_dim = model_dim // num_heads
        self.masked = masked
        self.chunk_len = chunk_len

        self.W_Q = nn.Linear(model_dim, model_dim, key=keys[0])
        self.W_K = nn.Linear(model_dim, model_dim, key=keys[1])
        self.W_V = nn.Linear(model_dim, model_dim, key=keys[2])
        self.W_O = nn.Linear(model_dim, model_dim, key=keys[3])

        if self.masked:
            self.tril = jnp.tril(jnp.ones((self.chunk_len, self.chunk_len)))

    def __call__(self, Q, K, V):
        seq_len, _ = Q.shape
        Q = jax.vmap(self.W_Q)(Q)
        K = jax.vmap(self.W_K)(K)
        V = jax.vmap(self.W_V)(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attn = self._scaled_dot_product_attention(Q, K, V)
        output = jax.vmap(self.W_O)(self._combine_heads(attn))
        return output

    def _split_heads(self, x):
        return x.reshape(self.chunk_len, self.num_heads, self.split_head_dim).transpose(1, 0, 2)

    def _combine_heads(self, x):
        return x.transpose(1, 0, 2).reshape(self.chunk_len, self.model_dim)

    def _scaled_dot_product_attention(self, Q, K, V):
        attention_scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / jnp.sqrt(self.split_head_dim)
        if self.masked:
            attention_scores = jnp.where(self.tril[:self.chunk_len, :self.chunk_len] == 0, float('-inf'), attention_scores)
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        output = jnp.matmul(attention_probs, V)
        return output
   
class AdaLNLayer(eqx.Module):
    attn: AdaLNAttention
    mlp: AdaLMLP
    adaLN_modulation: nn.Linear
    layer_norm1: nn.LayerNorm
    layer_norm2: nn.LayerNorm

    def __init__(self, model_dim, num_heads, chunk_len, dim_ff, dropout, masked, key=None):
        keys = jax.random.split(key, 3)
        self.attn = AdaLNAttention(model_dim, num_heads, chunk_len, masked, key=keys[0])
        self.mlp = AdaLMLP(model_dim, dim_ff, key=keys[1])
        self.adaLN_modulation = nn.Linear(model_dim, 6 * model_dim, use_bias=True, key=keys[2])

        self.layer_norm1 = nn.LayerNorm(model_dim, use_weight=False, use_bias=False, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(model_dim, use_weight=False, use_bias=False, eps=1e-6)

    def __call__(self, x, cond):
        modulated_cond = jax.nn.silu(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(jax.vmap(self.adaLN_modulation)(modulated_cond), 6, axis=-1)

        def modulate(x, shift, scale):
            return x * (1 + scale) + shift

        moduln = modulate(jax.vmap(self.layer_norm1)(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(moduln, moduln, moduln)

        moduln2 = modulate(jax.vmap(self.layer_norm2)(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * jax.vmap(self.mlp)(moduln2)
        return x

class FinalAdaLNLayer(eqx.Module):
    norm_final: nn.LayerNorm
    linear: nn.Linear
    adaLN_modulation: nn.Linear

    def __init__(self, model_dim, out_dim, key=None):
        keys = jax.random.split(key, 2)
        self.norm_final = nn.LayerNorm(model_dim, use_weight=False, use_bias=False, eps=1e-6)
        self.linear = nn.Linear(model_dim, out_dim, use_bias=True, key=keys[0])
        self.adaLN_modulation = nn.Linear(model_dim, 2 * model_dim, use_bias=True, key=keys[1])

    def __call__(self, x, c):
        modulated_c = jax.nn.silu(c)
        shift, scale = jnp.split(jax.vmap(self.adaLN_modulation)(modulated_c), 2, axis=-1)

        def modulate(x, shift, scale):
            return x * (1 + scale) + shift

        x = modulate(jax.vmap(self.norm_final)(x), shift, scale)
        x = jax.vmap(self.linear)(x)
        return x

class AdaLNTransformer(eqx.Module):
    layers: list
    final_layer: FinalAdaLNLayer
    config: dict = eqx.field(static=True)
    
    def __init__(self, config, out_dim: int, key=None):
        keys = jax.random.split(key, config['n_layers_actor']+ 1)
        self.config = config
        self.layers = [
            AdaLNLayer(
                model_dim=config['dim_ff'],
                num_heads=config['num_heads'],
                chunk_len=config['max_agents'],
                dim_ff=config['dim_ff'],
                dropout=config['dropout'],
                masked=config['masked'],
                key=keys[i]
            ) for i in range(config['n_layers_actor'])
        ]
        self.final_layer = FinalAdaLNLayer(config['dim_ff'], out_dim, key=keys[-1])

    def __call__(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return self.final_layer(x, cond)
    
class JaxTransformer(eqx.Module):
    state_enc: nn.Linear
    action_embedding: nn.Linear
    # NOTE: Positional embedding logic (SPE, SCE, RoPE) would be added here
    # For now, we'll use a simple one for demonstration.
    pos_embedding: Array
    decoder: AdaLNTransformer
    
    robot_indices: Array = eqx.field(static=True)
    is_critic: bool = eqx.field(static=True)

    def __init__(self, config, out_dim: int, is_critic: bool, key=None):
        keys = jax.random.split(key, 4)
        self.is_critic = is_critic
        self.robot_indices = jnp.arange(64)
        
        state_dim = config['state_dim']
        self.state_enc = nn.Linear(state_dim, config['dim_ff'], key=keys[0])
        self.action_embedding = nn.Linear(config['act_dim'], config['dim_ff'], key=keys[1])
        num_robots = 64
        embedding_dim = config['dim_ff']
        pos_embedding_model = SCEmbedding(num_robots, embedding_dim, key=keys[2])
        try:
            with open('./utils/jax/sce_model.eqx', 'rb') as f:
                pos_embedding_model = eqx.tree_deserialise_leaves(f, pos_embedding_model)
        except FileNotFoundError:
            print("WARNING: sce_model.eqx not found. Using randomly initialized positional embeddings.")
        
        arrays, static = eqx.partition(pos_embedding_model, eqx.is_array)
        arrays = jax.lax.stop_gradient(arrays)
        pos_embedding_model = eqx.combine(arrays, static)

        all_indices = jnp.arange(num_robots)
        self.pos_embedding = jax.vmap(pos_embedding_model)(all_indices)

        self.decoder = AdaLNTransformer(config, out_dim, key=keys[3])

    def __call__(self, state: Array, actions: Array = None, key: PRNGKeyArray = None):
        if not self.is_critic:
            n_agents = state.shape[0]
            initial_actions = jnp.zeros((n_agents, self.action_embedding.in_features))
            
            act_enc = jax.vmap(self.action_embedding)(initial_actions)
            state_enc = jax.vmap(self.state_enc)(state)
            pos_embed = self.pos_embedding
            
            conditional_enc = state_enc
            act_enc = act_enc + pos_embed
            logits = self.decoder(act_enc, conditional_enc)
            mu, log_std = jnp.split(logits, 2, axis=-1)
            return TanhNormal.from_loc_and_log_std(mu, log_std)

        else:
            if actions is None:
                raise ValueError("Critic must be provided with actions.")
            act_enc = jax.vmap(self.action_embedding)(actions)
            state_enc = jax.vmap(self.state_enc)(state)
            pos_embed = self.pos_embedding

            conditional_enc = state_enc
            act_enc = act_enc + pos_embed
            return self.decoder(act_enc, conditional_enc)