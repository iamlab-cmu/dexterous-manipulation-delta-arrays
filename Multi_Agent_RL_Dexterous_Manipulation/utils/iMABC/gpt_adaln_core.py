import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from einops import rearrange, reduce

from timm.models.vision_transformer import PatchEmbed

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

    def scaled_dot_product_attention(self, Q, K, V):
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.split_head_dim)
        if self.masked:
            attention_scores = attention_scores.masked_fill(self.tril[:self.n_agents, :self.n_agents] == 0, float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

    def split_heads(self, x):
        return x.view(self.bs, self.n_agents, self.num_heads, self.split_head_dim).transpose(1, 2)

    def combine_heads(self, x):
        return x.transpose(1, 2).contiguous().view(self.bs, self.n_agents, self.model_dim)

    def forward(self, Q, K, V):
        self.bs, self.n_agents, _ = Q.size()
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attn = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_O(self.combine_heads(attn))
        return output

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

# class IntegerEmbeddingModel(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim):
#         super(IntegerEmbeddingModel, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)

#     def forward(self, x):
#         return self.embedding(x)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class GPT_AdaLn_Block(nn.Module):
    """
    A GPT_AdaLn_ block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, num_heads, max_agents, masked=False)
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
    The final layer of GPT_AdaLn.
    """
    def __init__(self, model_dim, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.linear = wt_init_(nn.Linear(model_dim, action_dim, bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            wt_init_(nn.Linear(model_dim, 2 * model_dim, bias=True))
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class GPT_AdaLn_Actor(nn.Module):
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers):
        super(GPT_AdaLn_Actor, self).__init__()
        self.state_enc = wt_init_(nn.Linear(state_dim, model_dim))
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        # self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([GPT_AdaLn_Block(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        # self.final_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.final_layer = FinalLayer(model_dim, action_dim)
        # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos):
        state_enc = self.state_enc(state)
        obj_name_enc = self.obj_name_enc(obj_name_encs)
        pos_embed = self.pos_embedding(pos)

        conditional_enc = pos_embed.squeeze(2) + state_enc + obj_name_enc.unsqueeze(1)
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        output = self.final_layer(act_enc, conditional_enc)
        return output

class GPT_AdaLn_Critic(nn.Module):
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False):
        super(GPT_AdaLn_Critic, self).__init__()
        self.state_enc = wt_init_(nn.Linear(state_dim + action_dim, model_dim))
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.action_enc = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        # self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([GPT_AdaLn_Block(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        # self.final_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.final_layer = FinalLayer(model_dim, 1)
        # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos):
        # state_enc = self.state_enc(state)
        obj_name_enc = self.obj_name_enc(obj_name_encs)
        pos_embed = self.pos_embedding(pos)

        conditional_enc = pos_embed.squeeze(2) + obj_name_enc.unsqueeze(1)
        sa_enc = self.activation(torch.cat([self.action_enc(actions), self.state_enc(actions)], dim=-1))
        for layer in self.decoder_layers:
            sa_enc = layer(sa_enc, conditional_enc)
        output = self.final_layer(sa_enc, conditional_enc)
        return output

class Transformer(nn.Module):
    def __init__(self, hp_dict, delta_array_size = (8,8)):
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
        self.device = hp_dict['device']
        self.max_agents = delta_array_size[0] * delta_array_size[1]
        self.act_limit = hp_dict['act_limit']
        self.action_dim = hp_dict['action_dim']
        self.pos_embedding = IntegerEmbeddingModel(self.max_agents, hp_dict['model_dim'])
        self.pos_embedding.load_state_dict(torch.load(hp_dict['idx_embed_loc'], map_location=self.device, weights_only=True))
        for param in self.pos_embedding.parameters():
            param.requires_grad = False
        log_std = -0.5 * torch.ones(self.action_dim)
        self.log_std = torch.nn.Parameter(log_std)

        self.decoder_actor = GPT_AdaLn_Actor(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['decoder'])
        self.decoder_critic = GPT_AdaLn_Critic(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['decoder'])
        # self.decoder_critic2 = GPT_AdaLn_Critic(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['decoder'])
    
    def get_actions(self, states, obj_name_encs, pos, deterministic=False):
        """ Returns actor actions """
        bs, n_agents, _ = states.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)

        for i in range(n_agents):
            updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos)

            # TODO: Ablate here with all actions cloned so that even previous actions are updated with new info. 
            # TODO: Does it cause instability? How to know if it does?
            actions = actions.clone()
            actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
        return actions

    def compute_actor_loss(self, actions, states, obj_name_encs, pos):
        pred_actions = self.get_actions(states, obj_name_encs, pos)
        loss = F.mse_loss(actions, pred_actions)
        return loss
    
    def compute_critic_loss(self, actions, states, obj_name_encs, pos, rewards):
        # q1 = self.tf.decoder_critic1(states, actions, obj_name_encs, pos).mean(dim=1)
        q2 = self.tf.decoder_critic(states, actions, obj_name_encs, pos).mean(dim=1)
        loss = F.mse_loss(q2, rewards)
        return loss

class EMA:
    def __init__(self, ema_model, update_after_step, inv_gamma, power, min_value, max_value):
        self.ema_model = ema_model
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay=0
        self.optimization_step=0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)
        all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self.ema_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')

                if not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        self.optimization_step += 1