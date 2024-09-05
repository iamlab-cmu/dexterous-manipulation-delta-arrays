import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from einops import rearrange, reduce

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

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class GPTLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(GPTLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=True)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=True)
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

class GPT(nn.Module):
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False):
        super(GPT, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.critic = critic
        if self.critic:
            self.actor_mu_layer = wt_init_(nn.Linear(model_dim, 1))
        else:
            self.actor_mu_layer = wt_init_(nn.Linear(model_dim, action_dim))
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        pos_embed = self.pos_embedding(pos)
        obj_name_enc = self.obj_name_enc(obj_name_encs)

        conditional_enc = pos_embed.squeeze(2) + state_enc + obj_name_enc.unsqueeze(1)
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        act_mean = self.actor_mu_layer(act_enc)
        
        return act_mean
        # act_mean = self.actor_mu_layer(act_enc)
        # act_std = self.actor_std_layer(act_enc)
        # act_std = torch.clamp(act_std, LOG_STD_MIN, LOG_STD_MAX)
        # std = torch.exp(act_std)
        # return act_mean, std

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

        self.decoder_actor = GPT(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['decoder'])
    
    def get_actions(self, states, obj_name_encs, pos, deterministic=False):
        """ Returns actor actions """
        bs, n_agents, _ = states.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        for i in range(n_agents):
            updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos)

            # TODO: Ablate here with all actions cloned so that even previous actions are updated with new info. 
            # TODO: Does it cause instability? How to know if it does?
            actions = actions.clone()
            actions = self.act_limit * torch.tanh(updated_actions)
        return actions

    def compute_loss(self, actions, states, obj_name_encs, pos):
        bs, n_agents, _ = states.size()

        pred_actions = self.get_actions(states, obj_name_encs, pos)
        # loss = F.mse_loss(actions, pred_actions, reduction='none')
        # loss = reduce(loss, 'b ... -> b(...)', 'mean')
        # loss = loss.mean()

        loss = F.mse_loss(actions, pred_actions)
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