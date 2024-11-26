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
        super(EncoderLayer, self).__init__()
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
    
class Encoder(nn.Module):
    def __init__(self, state_dim, model_dim, num_heads, max_agents, dim_ff, dropout, n_layers):
        super(Encoder, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
        self.final_layer = wt_init_(nn.Linear(model_dim, model_dim))
        self.activation = nn.GELU()

    def forward(self, state):
        state_enc = self.activation(self.state_enc(state))
        for layer in self.encoder_layers:
            state_enc = layer(state_enc)
        return self.final_layer(state_enc)

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
        attn = self.cross_attention(encoder_output, x, x)
        x = self.layer_norm3(x + self.dropout(attn))
        ff_embed = self.feed_forward(x)
        x = x + self.dropout(ff_embed)
        return x

class GPT(nn.Module):
    def __init__(self, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False, masked=True):
        super(GPT, self).__init__()
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([GPTLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked) for _ in range(n_layers)])
        
        self.critic = critic
        if critic:
            self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
            self.final_layer = wt_init_(nn.Linear(model_dim, 1))
        else:
            self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
            self.final_layer_mu = wt_init_(nn.Linear(model_dim, action_dim))
            # self.final_layer_std = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state_enc, actions, pos, idx=None):
        pos_embed = self.pos_embedding(pos)
        conditional_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        
        if self.critic:
            q_val = self.final_layer(act_enc)
            return q_val
        else:
            mu = self.final_layer_mu(act_enc)
            # log_std = self.final_layer_std(act_enc)
            # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            # return mu, log_std
            return mu

# class RewardPositionVAE(nn.Module):
#     def __init__(self, n_agents, model_dim, latent_dim=32, hidden_dim=64):
#         """
#         VAE that learns to encode position and reward information.
        
#         Args:
#             n_agents (int): Number of agents
#             model_dim (int): Dimension to match transformer output
#             latent_dim (int): Size of the latent space
#             hidden_dim (int): Size of hidden layers
#         """
#         super().__init__()
#         self.n_agents = n_agents
#         self.model_dim = model_dim
#         self.latent_dim = latent_dim
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(n_agents + 1, hidden_dim),  # +1 for reward
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
        
#         # Mean and log variance projections
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, n_agents + 1)  # Reconstruct positions and reward
#         )
        
#         # Project latent vector to model dimension for each agent
#         self.reward_embedding = nn.Sequential(
#             nn.Linear(latent_dim, model_dim),
#             nn.LayerNorm(model_dim)
#         )
        
#     def encode(self, positions, reward):
#         """
#         Encode positions and reward into latent space.
        
#         Args:
#             positions (torch.Tensor): Shape (batch_size, n_agents, 1)
#             reward (torch.Tensor): Shape (batch_size, 1)
#         """
#         batch_size = positions.shape[0]
#         pos_flat = positions.view(batch_size, self.n_agents)
#         x = torch.cat([pos_flat, reward], dim=1)  # (batch_size, n_agents + 1)
        
#         x = self.encoder(x)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
        
#         return mu, logvar
    
#     def decode(self, z):
#         """Decode latent vector to positions and reward."""
#         return self.decoder(z)
    
#     def reparameterize(self, mu, logvar):
#         """Reparameterization trick."""
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def get_reward_aware_embedding(self, positions, reward):
#         """
#         Get reward-aware embeddings for each agent position.
        
#         Returns:
#             torch.Tensor: Shape (batch_size, n_agents, model_dim)
#         """
#         mu, logvar = self.encode(positions, reward)
#         z = self.reparameterize(mu, logvar)
        
#         # Project to model dimension and expand for each agent
#         reward_emb = self.reward_embedding(z).unsqueeze(1)  # (batch_size, 1, model_dim)
#         reward_emb = reward_emb.expand(-1, self.n_agents, -1)  # (batch_size, n_agents, model_dim)
        
#         return reward_emb, mu, logvar
    
#     def forward(self, positions, reward):
#         """Forward pass through VAE."""
#         mu, logvar = self.encode(positions, reward)
#         z = self.reparameterize(mu, logvar)
#         reconstruction = self.decode(z)
#         reward_aware_emb, mu, logvar = self.get_reward_aware_embedding(positions, reward)
        
#         return reconstruction, mu, logvar, reward_aware_emb

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
        
        self.encoder = Encoder(hp_dict['state_dim'], hp_dict['model_dim'], hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], hp_dict['dropout'], hp_dict['n_layers_dict']['encoder'])
        self.decoder_actor = GPT(hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], masked=hp_dict['masked'])
        self.decoder_critic1 = GPT(hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])
        self.decoder_critic2 = GPT(hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])

    # def forward(self, state, pos, idx=None):
    #     bs, n_agents, _ = state.size()
    #     mus, log_stds = torch.zeros((bs, n_agents, self.action_dim)).to(self.device), torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     mu, log_std = self.decoder_actor(self.encoder(state), torch.cat((mus, log_stds), dim=-1), pos, idx)
        
    #     std = log_std.exp()
    #     gauss_dist = torch.distributions.Normal(mu, std)
    #     actions = gauss_dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     logp_pi = gauss_dist.log_prob(actions).sum(axis=-1) - (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=-1)
    #     actions = torch.tanh(actions) * self.act_limit
    #     return actions, logp_pi, mu, std
    
        # y_t = torch.tanh(x_t)
        
        # action = y_t * self.act_limit
        # log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log(self.act_limit * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        # return action, log_prob
    
    def forward(self, state, pos, idx=None):
        bs, n_agents, _ = state.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        return self.act_limit * torch.tanh(self.decoder_actor(self.encoder(state), actions, pos, idx))

    @torch.no_grad()
    def get_actions(self, states,  pos, deterministic=False):
        """ Returns actor actions """
        bs, n_agents, _ = states.size()
        actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
        for i in range(n_agents):
            updated_actions = self.decoder_actor(self.encoder(states), actions, pos, i)
            actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
        return actions
    
    # @torch.no_grad()
    # def get_actions(self, state, pos, deterministic=False):
    #     """ Returns actor actions """
    #     bs, n_agents, _ = state.size()
    #     mus, log_stds = torch.zeros((bs, n_agents, self.action_dim)).to(self.device), torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     for i in range(n_agents):
    #         mu, log_std = self.decoder_actor(self.encoder(state), torch.cat((mus, log_stds), dim=-1), pos, i)
    #         mus[:, i, :], log_stds[:, i, :] = mu[:, i, :], log_std[:, i, :]
            
    #     if deterministic:
    #         actions = torch.tanh(mus) * self.act_limit
    #     else:
    #         std = log_std.exp()
    #         normal = torch.distributions.Normal(mu, std)
    #         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #         actions = torch.tanh(x_t) * self.act_limit
    #     return actions

    # def get_actions(self, states,  obj_name_encs, pos, deterministic=False):
    #     """ Returns actor actions """
    #     bs, n_agents, _ = states.size()
    #     actions = torch.zeros((bs, n_agents, self.action_dim)).to(self.device)
    #     for i in range(n_agents):
    #         updated_actions = self.decoder_actor(states, actions, obj_name_encs, pos, i)

    #         # TODO: Ablate here with all actions cloned so that even previous actions are updated with new info. 
    #         # TODO: Does it cause instability? How to know if it does?
    #         actions = actions.clone()
    #         actions[:, i, :] = self.act_limit * torch.tanh(updated_actions[:, i, :])
    #     return actions

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
