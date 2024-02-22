import numpy as np
import matplotlib.pyplot as plt

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

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ReLU(self.fc1(x)))

# Position Encoder for robotics should encode the ID of the robot (i,j)
# TODO: Change this function to capture the spatial arrangement of delta robots
class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_seq_len, model_dim, requires_grad=False)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class StateEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(StateEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=False)
        self.feed_forward = FF_MLP(model_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        attn = self.self_attention(x, x, x)
        x = self.layer_norm2(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        return x

class CriticDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(CriticDecoderLayer, self).__init__()
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=False)
        self.feed_forward = FF_MLP(model_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output):
        x = self.layer_norm1(x)
        attn = self.cross_attention(x, encoder_output, encoder_output)
        x = self.layer_norm2(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        return x

class ActorDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(ActorDecoderLayer, self).__init__()
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

class StateEncoder(nn.Module):
    def __init__(self, model_dim, state_dim, num_heads, max_agents, dim_ff, dropout, n_layers, pos_enc):
        super(StateEncoder, self).__init__()
        self.state_embedding = wt_init_(nn.Linear(state_dim, model_dim))
        self.positional_encoding = pos_enc
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([StateEncoderLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.ReLU = nn.ReLU()

    def forward(self, states):
        """
        Input: states (bs, n_agents, state_dim)
        Output: state_enc (bs, n_agents, model_dim)
        """
        state_enc = self.dropout(self.positional_encoding(self.ReLU(self.state_embedding(states))))
        for layer in self.encoder_layers:
            state_enc = layer(state_enc)
        return state_enc

class CriticDecoder(nn.Module):
    def __init__(self, model_dim, action_dim, num_heads, max_agents, dim_ff, dropout, n_layers, pos_enc):
        super(CriticDecoder, self).__init__()
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.positional_encoding = pos_enc
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([CriticDecoderLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.critic_op_layer = wt_init_(nn.Linear(model_dim, 1))
        self.ReLU = nn.ReLU()

    def forward(self, state_enc, actions):
        """
        Input: state_enc (bs, n_agents, model_dim)
               actions (bs, n_agents, action_dim)
        Output: q_value (bs, 1) --> Centralized Critic Q' = 1/N * ∑Q
        """
        act_enc = self.dropout(self.positional_encoding(self.ReLU(self.action_embedding(actions))))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, state_enc)
        q_val = self.critic_op_layer(act_enc)
        return torch.mean(q_val, axis=1)

class ActorDecoder(nn.Module):
    def __init__(self, model_dim, action_dim, num_heads, max_agents, dim_ff, dropout, n_layers, pos_enc):
        super(ActorDecoder, self).__init__()
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim)) # Replace action embedding of Critic Decoder from this.
        self.positional_encoding = pos_enc
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([ActorDecoderLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.actor_mu_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.ReLU = nn.ReLU()

    def forward(self, state_enc, actions):
        """
        Input: state_enc (bs, n_agents, model_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        act_enc = self.ReLU(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, state_enc)
        act_mean = self.actor_mu_layer(act_enc)
        return act_mean
        # act_mean = self.actor_mu_layer(act_enc)
        # act_std = self.actor_std_layer(act_enc)
        # act_std = torch.clamp(act_std, LOG_STD_MIN, LOG_STD_MAX)
        # std = torch.exp(act_std)
        # return act_mean, std

class Transformer(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit, model_dim, num_heads, dim_ff, num_layers, dropout, device, delta_array_size = (8,8)):
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
        self.device = device
        self.max_agents = delta_array_size[0] * delta_array_size[1]
        self.act_limit = action_limit
        self.action_dim = action_dim
        self.positional_encoding = PositionalEncoder(model_dim, self.max_agents)

        self.encoder = StateEncoder(model_dim, state_dim, num_heads, self.max_agents, dim_ff, dropout, num_layers['encoder'], self.positional_encoding)
        self.decoder_critic1 = CriticDecoder(model_dim, action_dim, num_heads, self.max_agents, dim_ff, dropout, num_layers['critic'], self.positional_encoding)
        self.decoder_critic2 = CriticDecoder(model_dim, action_dim, num_heads, self.max_agents, dim_ff, dropout, num_layers['critic'], self.positional_encoding)
        
        self.decoder_actor = ActorDecoder(model_dim, action_dim, num_heads, self.max_agents, dim_ff, dropout, num_layers['actor'], self.positional_encoding)

    def eval_actor(self, state_enc, actions):
        """
        Input: state_enc (bs, n_agents, model_dim)
        Output: actions (bs, n_agents, action_dim)
        """
        shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        shifted_actions[:, 1:, :] = actions[:, :-1, :]  # Input actions go from 0 to A_n-1
        act_mean, act_std = self.decoder_actor(state_enc, shifted_actions)
        
        dist = torch.distributions.Normal(act_mean, act_std)
        entropy = dist.entropy()

        output_actions = dist.rsample()
        log_probs = dist.log_prob(output_actions).sum(axis=-1)
        log_probs -= (2*(np.log(2) - output_actions - F.softplus(-2*output_actions))).sum(axis=2)
        
        output_actions = torch.tanh(output_actions)
        output_actions = self.act_limit * output_actions # Output actions go from A_0 to A_n
        return output_actions, log_probs, entropy

    def get_actions_lin(self, state_enc, deterministic=False):
        """ Returns actor actions, and their log probs. If deterministic=True, set action as the output of decoder. Else, sample from mean=dec output, std=exp(log_std) """
        bs, n_agents, _ = state_enc.size()
        shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        output_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)

        for i in range(n_agents):
            act_encs = self.decoder_actor(state_enc, shifted_actions)
            act = act_encs[:, i, :]

            output_action = self.act_limit * torch.tanh(act)
            output_actions = output_actions.clone()
            output_actions[:, i, :] = output_action
            
            if (i+1) < n_agents:
                shifted_actions = shifted_actions.clone()
                shifted_actions[:, i+1, :] = output_action
        return output_actions


    def get_actions_gauss(self, state_enc, deterministic=False):
        """ Returns actor actions, and their log probs. If deterministic=True, set action as the output of decoder. Else, sample from mean=dec output, std=exp(log_std) """
        bs, n_agents, _ = state_enc.size()
        shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        output_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        output_action_log_probs = torch.zeros((bs, n_agents)).to(self.device)

        

        for i in range(n_agents):
            # act_means, act_stds = self.decoder_actor(state_enc, shifted_actions)
            
            # dist = torch.distributions.Normal(act_means, act_stds)
            # output_actions = dist.rsample()

            # output_action_log_probs = dist.log_prob(output_actions).sum(axis=-1)
            # output_action_log_probs = output_action_log_probs - (2*(np.log(2) - output_actions - F.softplus(-2*output_actions))).sum(axis=2)
            
            # output_actions = self.act_limit * torch.tanh(output_actions)
            act_means, act_stds = self.decoder_actor(state_enc, shifted_actions)
            mean, std = act_means[:, i, :], act_stds[:, i, :]

            if deterministic:
                output_action = self.act_limit * torch.tanh(mean)
                output_action_log_prob = 0
            else:
                dist = torch.distributions.Normal(mean, std)
                output_action = dist.rsample()

                output_action_log_prob = dist.log_prob(output_action).sum(axis=-1)
                output_action_log_prob = output_action_log_prob - (2*(np.log(2) - output_action - F.softplus(-2*output_action))).sum(axis=1)
                
                output_action = self.act_limit * torch.tanh(output_action)

            output_actions = output_actions.clone()
            output_actions[:, i, :] = output_action
            output_action_log_probs = output_action_log_probs.clone()
            output_action_log_probs[:, i] = output_action_log_prob
            if (i+1) < n_agents:
                shifted_actions = shifted_actions.clone()
                shifted_actions[:, i+1, :] = output_action
        return output_actions, output_action_log_probs