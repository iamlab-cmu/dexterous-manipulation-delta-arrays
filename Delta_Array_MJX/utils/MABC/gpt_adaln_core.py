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

    @torch.no_grad()
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

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

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

class GPT_AdaLN(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers, critic=False, masked=True, gauss=False):
        super(GPT_AdaLN, self).__init__()
        self.state_enc = nn.Linear(state_dim, model_dim)
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([AdaLNLayer(model_dim, num_heads, max_agents, dim_ff, dropout, masked) for _ in range(n_layers)])
        self.critic = critic
        self.gauss = gauss
        if self.critic:
            self.final_layer = FinalLayer(model_dim, 1)
        else:
            if self.gauss:
                self.mu = FinalLayer(model_dim, action_dim)
                self.log_std = FinalLayer(model_dim, action_dim)
            else:
                self.final_layer = FinalLayer(model_dim, action_dim)
            # self.actor_std_layer = wt_init_(nn.Linear(model_dim, action_dim))
        self.activation = nn.GELU()

    def forward(self, state, actions, pos, idx=None):
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

class Transformer(nn.Module):
    def __init__(self, hp_dict, delta_array_size = (8,8)):
        super(Transformer, self).__init__()
        self.hp_dict = hp_dict
        self.device = hp_dict['device']
        self.max_agents = delta_array_size[0] * delta_array_size[1]
        self.act_limit = hp_dict['act_limit']
        self.action_dim = hp_dict['action_dim']
        self.pos_embedding = IntegerEmbeddingModel(self.max_agents, hp_dict['model_dim'])
        self.pos_embedding.load_state_dict(torch.load(hp_dict['idx_embed_loc'], map_location=self.device, weights_only=True))
        for param in self.pos_embedding.parameters():
            param.requires_grad = False
        self.gauss = hp_dict['gauss']

        self.decoder_actor = GPT_AdaLN(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'], masked=hp_dict['masked'], gauss=hp_dict['gauss'])
        self.decoder_critic = GPT_AdaLN(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'], critic=True, masked=hp_dict['masked'])
        self.alpha = 0.2
    
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

    def compute_actor_loss(self, actions, states, pos):
        if self.gauss:
            pred_actions, _, _, _ = self.forward(states, pos)
        else:
            pred_actions = self.forward(states, pos)
        loss = F.mse_loss(actions, pred_actions)
        return loss
    
    def compute_critic_loss(self, s1, a, s2, pos, rewards, d):
        q = self.decoder_critic(s1, a, pos).squeeze().mean(dim=1)
        
        with torch.no_grad():
            if self.gauss:
                next_actions, log_probs, _, _= self.forward(s2, pos)
            else:
                next_actions = self.forward(s2, pos)
            
            next_q = self.decoder_critic(s2, next_actions, pos).squeeze()
            q_next = rewards + self.hp_dict['gamma'] * ((1 - d.unsqueeze(1)) * (next_q - self.alpha * log_probs)).mean(dim=1)

        loss = F.mse_loss(q, q_next)
        return loss
    
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