import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def wt_init_(l, activation = "relu"):
    nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain(activation))
    if l.bias is not None:
        nn.init.constant_(l.bias, 0)
    return l

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def build_rotary_pos_emb(seq_len, dim, base=10000.0):
    half_dim = dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=torch.float32)  
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))  # (half_dim,)

    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    angles = t * inv_freq.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    bs, n_heads, seq_len, dim = q.shape
    half_dim = dim // 2
    
    cos = cos[offset: offset + seq_len, :].to(q.device)
    sin = sin[offset: offset + seq_len, :].to(q.device)

    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    cos_ = cos.unsqueeze(0).unsqueeze(0)  
    sin_ = sin.unsqueeze(0).unsqueeze(0)
    
    q_rotated = torch.cat([q1 * cos_ - q2 * sin_, q1 * sin_ + q2 * cos_], dim=-1)
    k_rotated = torch.cat([k1 * cos_ - k2 * sin_, k1 * sin_ + k2 * cos_], dim=-1)
    return q_rotated, k_rotated

class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, masked):
        super(MultiHeadAttentionRoPE, self).__init__()
        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.split_head_dim = model_dim // num_heads
        self.masked = masked
        self.max_agents = max_agents

        self.W_Q = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_K = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_V = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_O = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")

        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(max_agents, max_agents)))

        # --- RoPE additions ---
        # We'll build cos and sin up to max_agents length
        cos, sin = build_rotary_pos_emb(seq_len=max_agents, dim=self.split_head_dim)
        self.register_buffer("rope_cos", cos)  # shape => (max_agents, split_head_dim//2)
        self.register_buffer("rope_sin", sin)  # shape => (max_agents, split_head_dim//2)

    def scaled_dot_product_attention(self, Q, K, V, n_agents):
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.split_head_dim)
        if self.masked:
            attention_scores = attention_scores.masked_fill(
                self.tril[:n_agents, :n_agents] == 0, float('-inf')
            )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

    def split_heads(self, x, bs, n_agents):
        # from (bs, n_agents, model_dim) -> (bs, num_heads, n_agents, split_head_dim)
        return x.view(bs, n_agents, self.num_heads, self.split_head_dim).transpose(1, 2)

    def combine_heads(self, x, bs, n_agents):
        # from (bs, num_heads, n_agents, split_head_dim) -> (bs, n_agents, model_dim)
        return x.transpose(1, 2).contiguous().view(bs, n_agents, self.model_dim)

    def forward(self, Q, K, V):
        bs, n_agents, _ = Q.size()
        Q = self.split_heads(self.W_Q(Q), bs, n_agents)
        K = self.split_heads(self.W_K(K), bs, n_agents)
        V = self.split_heads(self.W_V(V), bs, n_agents)

        # --------------- RoPE step ---------------
        # Q, K: (bs, num_heads, n_agents, split_head_dim)
        Q, K = apply_rotary_pos_emb(Q, K, self.rope_cos, self.rope_sin, offset=0)
        # -----------------------------------------

        attn = self.scaled_dot_product_attention(Q, K, V, n_agents)
        output = self.W_O(self.combine_heads(attn, bs, n_agents))
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, masked):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.split_head_dim = model_dim // num_heads
        self.masked = masked

        self.W_Q = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_K = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_V = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_O = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")

        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(max_agents, max_agents)))

    def scaled_dot_product_attention(self, Q, K, V, n_agents):
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.split_head_dim)
        if self.masked:
            attention_scores = attention_scores.masked_fill(self.tril[:n_agents, :n_agents] == 0, float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

    def split_heads(self, x, bs, n_agents):
        return x.view(bs, n_agents, self.num_heads, self.split_head_dim).transpose(1, 2)

    def combine_heads(self, x, bs, n_agents):
        return x.transpose(1, 2).contiguous().view(bs, n_agents, self.model_dim)

    def forward(self, Q, K, V):
        bs, n_agents, _ = Q.size()
        Q = self.split_heads(self.W_Q(Q), bs, n_agents)
        K = self.split_heads(self.W_K(K), bs, n_agents)
        V = self.split_heads(self.W_V(V), bs, n_agents)

        attn = self.scaled_dot_product_attention(Q, K, V, n_agents)
        output = self.W_O(self.combine_heads(attn, bs, n_agents))
        return output

class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(SinusoidalPosEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # shape: [max_len, d_model]

    def forward(self, pos):
        pos = pos.squeeze(-1)
        pe_out = self.pe[pos, :]
        pe_out = pe_out.unsqueeze(2)
        return pe_out

class IntegerEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(IntegerEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim) 

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class GPTLayer_CA(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha):
        super(GPTLayer_CA, self).__init__()
        self.self_attention = mha(model_dim, num_heads, max_agents, masked=masked)
        self.cross_attention = mha(model_dim, num_heads, max_agents, masked=masked)
        self.feed_forward = FF_MLP(model_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output):
        x = self.layer_norm1(x)
        attn = self.self_attention(x, x, x)
        x = self.layer_norm2(x + self.dropout(attn))
        attn = self.cross_attention(x, encoder_output, encoder_output)
        x = self.layer_norm3(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        return x
    
class GPTLayer_SA(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha):
        super(GPTLayer_SA, self).__init__()
        self.self_attention = mha(2*model_dim, num_heads, max_agents, masked=masked)
        self.feed_forward = FF_MLP(2*model_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(2*model_dim)
        self.layer_norm2 = nn.LayerNorm(2*model_dim)
        self.fc = wt_init_(nn.Linear(2*model_dim, model_dim))

    def forward(self, act_enc, conditional_enc):
        x = torch.cat([act_enc, conditional_enc], dim=-1)
        x = self.layer_norm1(x)
        attn = self.self_attention(x, x, x)
        x = self.layer_norm2(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        x = self.fc(x)
        return x

class GPT(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, mha, critic=False, masked=True, gauss=False, SA=False):
        super(GPT, self).__init__()
        self.SA = SA
        print(model_dim)
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        if self.SA:
            self.decoder_layers = nn.ModuleList([GPTLayer_SA(model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha) for _ in range(n_layers)])
        else:
            self.decoder_layers = nn.ModuleList([GPTLayer_CA(model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha) for _ in range(n_layers)])
        self.critic = critic
        self.gauss = gauss
        if self.critic:
            self.final_layer = wt_init_(nn.Linear(model_dim, 1))
        else:
            if self.gauss:
                self.mu = wt_init_(nn.Linear(model_dim, action_dim))
                self.log_std = wt_init_(nn.Linear(model_dim, action_dim))
            else:
                self.final_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, pos, idx=None):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        pos_embed = self.pos_embedding(pos).squeeze(2)
        
        conditional_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        
        if self.gauss:
            act_mean = self.mu(act_enc)
            act_std = self.log_std(act_enc)
            act_std = torch.clamp(act_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(act_std)
            return act_mean, std
        else:
            act_mean = self.final_layer(act_enc)
        return act_mean

class AdaLNLayer(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha):
        super().__init__()
        self.attn = mha(model_dim, num_heads, max_agents, masked=masked)
        self.mlp = FF_MLP(model_dim, dim_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim, bias=True)
        )
        self.layer_norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
        moduln = modulate(self.layer_norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(moduln, moduln, moduln)
        x = x + gate_mlp * self.mlp(modulate(self.layer_norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_dim, action_dim, critic=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.linear = wt_init_(nn.Linear(model_dim, action_dim, bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            wt_init_(nn.Linear(model_dim, 2 * model_dim, bias=True))
        )
        self.critic = critic

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        if self.critic:
            x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        x = self.linear(x)
        return x

class GPT_AdaLN(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, mha, critic=False, masked=True, gauss=False):
        super(GPT_AdaLN, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        if pos_embedding is not None:
            self.fwd_fn = self.forward_normal
        else:
            self.fwd_fn = self.forward_rotpe
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([AdaLNLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked, mha) for _ in range(n_layers)])
        self.critic = critic
        self.gauss = gauss
        if self.critic:
            self.final_layer = FinalLayer(model_dim, 1, critic=True)
        else:
            if self.gauss:
                self.mu = FinalLayer(model_dim, action_dim)
                self.log_std = FinalLayer(model_dim, action_dim)
            else:
                self.final_layer = FinalLayer(model_dim, action_dim)
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, pos, idx=None):
        return self.fwd_fn(state, actions, pos, idx)

    def forward_normal(self, state, actions, pos, idx=None):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        pos_embed = self.pos_embedding(pos)
                                            
        conditional_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
            
        if self.gauss:
            act_mean = self.mu(act_enc, conditional_enc)
            act_std = self.log_std(act_enc, conditional_enc)
            act_std = torch.clamp(act_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(act_std)
            return act_mean, std
        else:
            act_mean = self.final_layer(act_enc, conditional_enc)
        return act_mean

    def forward_rotpe(self, state, actions, pos, idx=None):
        state_enc = self.state_enc(state)
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, state_enc)
            
        if self.gauss:
            act_mean = self.mu(act_enc, state_enc)
            act_std = self.log_std(act_enc, state_enc)
            act_std = torch.clamp(act_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(act_std)
            return act_mean, std
        else:
            act_mean = self.final_layer(act_enc, state_enc)
        return act_mean

class Transformer(nn.Module):
    def __init__(self, hp_dict, delta_array_size = (8, 8)):
        super(Transformer, self).__init__()
        """
        For 2D planar manipulation:
            state_dim = 6: state, goal, pos of robot
            action_dim = 2: action of robot
        max_agents = delta_array_size[0] * delta_array_size[1]: Maximum number of agents in the environment
        model_dim: size of attn layers (model_dim % num_heads = 0)
        dim_ff: size of MLPs
        num_layers: number of layers in encoder and decoder
        """
        self.hp_dict = hp_dict
        self.device = hp_dict['dev_rl']
        self.max_agents = delta_array_size[0] * delta_array_size[1]
        self.act_limit = hp_dict['act_limit']
        self.action_dim = hp_dict['action_dim']
        mha = None
        if hp_dict['pos_embed'] == "SPE":
            self.pos_embedding = SinusoidalPosEmbedding(
                d_model=256,
                max_len=self.max_agents
            )
            mha = MultiHeadAttention
            
        elif hp_dict['pos_embed'] == "SCE":
            self.pos_embedding = IntegerEmbeddingModel(self.max_agents, embedding_dim=256)
            self.pos_embedding.load_state_dict(
                torch.load("./utils/MATSAC/idx_embedding_new.pth",
                           map_location=self.device,
                           weights_only=True)
            )
            for param in self.pos_embedding.parameters():
                param.requires_grad = False
            mha = MultiHeadAttention
            
        elif hp_dict['pos_embed'] == "RoPE":
            self.pos_embedding = None
            mha = MultiHeadAttentionRoPE
            
        if mha is None:
            raise ValueError("pos_embed must be one of ['SPE', 'SCE', 'RoPE']")
        
        log_std = -0.5 * torch.ones(self.action_dim)
        self.log_std = torch.nn.Parameter(log_std)
        self.gauss = hp_dict['gauss']

        if hp_dict["attn_mech"] == "AdaLN":
            self.decoder_actor = GPT_AdaLN(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], mha, masked=hp_dict['masked'], gauss=hp_dict['gauss'])
            self.decoder_critic = GPT_AdaLN(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], mha, critic=True, masked=hp_dict['masked'])
        elif hp_dict["attn_mech"] in ["CA", "SA"]:
            sa_bool = False
            if hp_dict["attn_mech"] == "SA":
                sa_bool = True
            self.decoder_actor = GPT(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], mha, masked=hp_dict['masked'], gauss=hp_dict['gauss'], SA=sa_bool)
            self.decoder_critic = GPT(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], mha, critic=True, masked=hp_dict['masked'], SA=sa_bool)

    def forward(self, state, pos):
        bs, n_agents, _ = state.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        if self.gauss:
            mu, std = self.decoder_actor(state, actions, pos)
            
            gauss_dist = torch.distributions.Normal(mu, std)
            actions = gauss_dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
            logp_pi = gauss_dist.log_prob(actions).sum(axis=-1) - (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=-1)
            actions = torch.tanh(actions) * self.act_limit
            return actions, logp_pi, mu, std
        else:
            return self.act_limit * torch.tanh(self.decoder_actor(state, actions, pos))

    @torch.no_grad()
    def get_actions(self, state,  pos, deterministic=False):
        bs, n_agents, _ = state.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        if self.gauss:
            mu, std = self.decoder_actor(state, actions, pos)
            if deterministic:
                return torch.tanh(mu) * self.act_limit
            
            gauss_dist = torch.distributions.Normal(mu, std)
            actions = gauss_dist.rsample()
            return torch.tanh(actions) * self.act_limit
        else:
            return self.act_limit * torch.tanh(self.decoder_actor(state, actions, pos))