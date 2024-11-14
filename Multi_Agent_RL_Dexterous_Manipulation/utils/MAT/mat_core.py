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

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(GPTLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, max_agents)
        self.feed_forward = FF_MLP(model_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output):
        x = self.layer_norm1(x)
        attn = self.self_attention(x, x, x)
        x = self.layer_norm2(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        return x
    
class GPT(nn.Module):
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False, masked=True):
        super(GPT, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked) for _ in range(n_layers)])
        self.critic = critic
        self.actor_mu_layer = wt_init_(nn.Linear(model_dim, 1))
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos, idx=None):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        obj_name_enc = self.obj_name_enc(obj_name_encs)
        pos_embed = self.pos_embedding(pos)
        
        conditional_enc = pos_embed.squeeze(2) + state_enc + obj_name_enc.unsqueeze(1)
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        act_mean = self.actor_mu_layer(act_enc)
        
        return act_mean

class GPTLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout, masked):
        super(GPTLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=masked)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, max_agents, masked=masked)
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
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False, masked=True):
        super(GPT, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked) for _ in range(n_layers)])
        self.critic = critic
        if self.critic:
            self.actor_mu_layer = wt_init_(nn.Linear(model_dim, 1))
        else:
            self.actor_mu_layer = wt_init_(nn.Linear(model_dim, action_dim))
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos, idx=None):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        obj_name_enc = self.obj_name_enc(obj_name_encs)
        pos_embed = self.pos_embedding(pos)
        
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

class AdaLNLayer(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout, masked):
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, num_heads, max_agents, masked=masked)
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


class GPT_AdaLN(nn.Module):
    def __init__(self, state_dim, obj_name_enc_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False, masked=True):
        super(GPT_AdaLN, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.obj_name_enc = nn.Embedding(obj_name_enc_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([AdaLNLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked) for _ in range(n_layers)])
        self.critic = critic
        if self.critic:
            self.final_layer = FinalLayer(model_dim, 1)
        else:
            self.final_layer = FinalLayer(model_dim, action_dim)
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, obj_name_encs, pos, idx=None):
        """
        Input: state (bs, n_agents, state_dim)
               actions (bs, n_agents, action_dim)
        Output: decoder_output (bs, n_agents, model_dim)
        """
        # act_enc = self.dropout(self.positional_encoding(F.ReLU(self.action_embedding(actions))))
        state_enc = self.state_enc(state)
        pos_embed = self.pos_embedding(pos)
        obj_name_enc = self.obj_name_enc(obj_name_encs)
        
        conditional_enc = pos_embed.squeeze(2) + obj_name_enc.unsqueeze(1) + state_enc
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        act_mean = self.final_layer(act_enc, conditional_enc)
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
        self.pos_embedding = IntegerEmbeddingModel(self.max_agents, embedding_dim=256)
        self.pos_embedding.load_state_dict(torch.load("./utils/MATSAC/idx_embedding_new.pth", map_location=self.device, weights_only=True))
        for param in self.pos_embedding.parameters():
            param.requires_grad = False
        log_std = -0.5 * torch.ones(self.action_dim)
        self.log_std = torch.nn.Parameter(log_std)

        if hp_dict["adaln"]:
            self.decoder_actor = GPT_AdaLN(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], masked=hp_dict['masked'])
            self.decoder_critic1 = GPT_AdaLN(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])
            self.decoder_critic2 = GPT_AdaLN(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])
        else:
            self.decoder_actor = GPT(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], masked=hp_dict['masked'])
            self.decoder_critic1 = GPT(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])
            self.decoder_critic2 = GPT(hp_dict['state_dim'], hp_dict['obj_name_enc_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])

    def get_actions(self, states,  obj_name_encs, pos, deterministic=False):
        """ Returns actor actions """
        bs, n_agents, _ = states.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        for i in range(n_agents):
            updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos, i)

            # TODO: Ablate here with all actions cloned so that even previous actions are updated with new info. 
            # TODO: Does it cause instability? How to know if it does?
            actions = actions.clone()
            actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
        return actions

    # def get_actions(self, states, pos, deterministic=False):
    #     """ Returns actor actions, and their log probs. If deterministic=True, set action as the output of decoder. Else, sample from mean=dec output, std=exp(log_std) """
    #     bs, n_agents, _ = states.size()
    #     shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     output_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     output_action_log_probs = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)

    #     for i in range(n_agents):            
    #         act_mean, std = self.decoder_actor(states, shifted_actions, pos, i)
    #         # std = torch.sigmoid(self.log_std)*0.5

    #         if deterministic:
    #             output_action = self.act_limit * torch.tanh(act_mean)
    #             output_action_log_prob = 0
    #         else:
    #             dist = torch.distributions.Normal(act_mean, std)
    #             output_action = dist.rsample()

    #             output_action_log_prob = dist.log_prob(output_action).sum(axis=-1)
    #             output_action_log_prob = output_action_log_prob - (2*(np.log(2) - output_action - F.softplus(-2*output_action))).sum(axis=1)
                
    #             output_action = self.act_limit * torch.tanh(output_action)

            
    #         output_actions = output_actions.clone()
    #         output_actions[:, i, :] = output_action
    #         output_action_log_probs = output_action_log_probs.clone()
    #         output_action_log_probs[:, i, :] = output_action_log_prob

    #         if (i+1) < n_agents:
    #             shifted_actions = shifted_actions.clone()
    #             shifted_actions[:, i+1, :] = output_action
    #     return output_actions, output_action_log_probs


    # def get_action_values(self, states, actions):
    #     """
    #     Input: state_enc (bs, n_agents, model_dim)
    #     Output: actions (bs, n_agents, action_dim)
    #     """
    #     state_enc = self.state_enc_critic(states)
    #     shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     shifted_actions[:, 1:, :] = actions[:, :-1, :]  # Input actions go from 0 to A_n-1
    #     q_vals = self.decoder_critic(state_enc, shifted_actions)
    #     return q_vals.squeeze().mean()

    def eval_actor_gauss(self, states, actions):
        """
        Input: state_enc (bs, n_agents, model_dim)
        Output: actions (bs, n_agents, action_dim)
        """
        state_enc = self.state_enc(states)
        shifted_actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        shifted_actions[:, 1:, :] = actions[:, :-1, :]  # Input actions go from 0 to A_n-1
        act_mean, act_std = self.decoder_critic(state_enc, shifted_actions)
        
        dist = torch.distributions.Normal(act_mean, act_std)
        entropy = dist.entropy()

        output_actions = dist.rsample()
        log_probs = dist.log_prob(output_actions).sum(axis=-1)
        log_probs -= (2*(np.log(2) - output_actions - F.softplus(-2*output_actions))).sum(axis=2)
        
        output_actions = torch.tanh(output_actions)
        output_actions = self.act_limit * output_actions # Output actions go from A_0 to A_n
        return output_actions, log_probs, entropy
