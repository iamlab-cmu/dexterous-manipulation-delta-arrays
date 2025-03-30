import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange, reduce
from torch.optim.lr_scheduler import _LRScheduler

LOG_STD_MAX = 2
LOG_STD_MIN = -20

""" Following 3 functions are taken from DiT github repo. """
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        betas = np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
        return np.array(betas, dtype=np.float64)
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

def wt_init_(l, activation = "relu"):
    nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain(activation))
    if l.bias is not None:
        nn.init.constant_(l.bias, 0)
    return l

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# def CAWR_with_Warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles = 0.5, last_epoch = -1
#     ) -> LambdaLR:

#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
#     return LambdaLR(optimizer, lr_lambda, last_epoch)


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

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class DiTLayer(nn.Module):
    def __init__(self, model_dim, num_heads, max_agents, dim_ff, dropout):
        super(DiTLayer, self).__init__()
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

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
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

class DiTActor(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers):
        super(DiTActor, self).__init__()
        self.state_enc = wt_init_(nn.Linear(state_dim, model_dim))
        self.action_embedding = wt_init_(nn.Linear(action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([DiTBlock(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
            
        self.final_layer = FinalLayer(model_dim, action_dim)
        self.activation = nn.SiLU()
        
    def forward(self, state, actions, pos):
        state_enc = self.state_enc(state)
        pos_embed = self.pos_embedding(pos)

        conditional_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.action_embedding(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, conditional_enc)
        pred_noise = self.final_layer(act_enc, conditional_enc)
        return pred_noise

class DiTCritic(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers):
        super(DiTCritic, self).__init__()
        self.state_action_enc = wt_init_(nn.Linear(state_dim + action_dim, model_dim))
        self.q_enc = wt_init_(nn.Linear(1, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DiTBlock(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
            
        self.final_layer = FinalLayer(model_dim, 1)
        self.activation = nn.SiLU()
        
    def forward(self, states, actions, pos, q_values=None):
        SA = torch.cat((states, actions), dim=-1)
        state_action_enc = self.state_action_enc(SA)
        pos_embed = self.pos_embedding(pos)

        conditional_enc = pos_embed.squeeze(2) + state_action_enc
        q_enc = self.activation(self.q_enc(q_values))
        for layer in self.decoder_layers:
            q_enc = layer(q_enc, conditional_enc)
        pred_noise = self.final_layer(q_enc, conditional_enc)
        return pred_noise
    
class OGCritic(nn.Module):
    def __init__(self, state_dim, model_dim, action_dim, num_heads, max_agents, dim_ff, pos_embedding, dropout, n_layers):
        super(OGCritic, self).__init__()
        self.state_action_enc = wt_init_(nn.Linear(state_dim + action_dim, model_dim))
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DiTBlock(model_dim, num_heads, max_agents, dim_ff, dropout) for _ in range(n_layers)])
            
        self.final_layer = FinalLayer(model_dim, 1)
        self.activation = nn.SiLU()
        
    def forward(self, states, actions, pos, q_values=None):
        SA = torch.cat((states, actions), dim=-1)
        state_action_enc = self.state_action_enc(SA)
        pos_embed = self.pos_embedding(pos).squeeze(2)

        q_enc = self.activation(state_action_enc)
        for layer in self.decoder_layers:
            q_enc = layer(q_enc, pos_embed)
        pred_noise = self.final_layer(q_enc, pos_embed)
        return pred_noise


class DiffusionTransformer(nn.Module):
    def __init__(self, hp_dict, expt="expt_1", delta_array_size = (8,8)):
        super(DiffusionTransformer, self).__init__()
        """
        For 2D planar manipulation:
            state_dim = 6: state, goal, pos of robot
            action_dim = 2: action of robot
        max_agents = delta_array_size[0] * delta_array_size[1]: Maximum number of agents in the environment
        model_dim: size of attn layers (model_dim % num_heads = 0)
        dim_ff: size of MLPs
        num_layers: number of layers in encoder and decoder
        """
        self.device = hp_dict['dev_rl']
        self.max_agents = delta_array_size[0] * delta_array_size[1]
        self.action_dim = hp_dict['action_dim']
        self.act_lim_low = hp_dict['env_dict']['action_space']['low']
        self.act_lim_high = hp_dict['env_dict']['action_space']['high']
        self.pos_embedding = IntegerEmbeddingModel(self.max_agents, hp_dict['model_dim'])
        self.pos_embedding.load_state_dict(torch.load(hp_dict['idx_embed_loc'], map_location=self.device, weights_only=True))
        for param in self.pos_embedding.parameters():
            param.requires_grad = False
        self.entropy = hp_dict['entropy']

        self.denoising_params = hp_dict['denoising_params']
        self.diff_timesteps = self.denoising_params['n_diff_steps']

        # Below diffusion coefficients and posterior variables copied from DiT git repo
        self.betas = get_named_beta_schedule(self.denoising_params['beta_schedule'], self.diff_timesteps)
        # self.betas = get_named_beta_schedule('linear', 1000)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # For Var Type: Fixed small log calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = np.maximum((self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)), hp_dict['exp_noise']**2)
        
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        self.posterior_mean_coef1 = (self.betas * np.sqrt(self.alphas_cumprod_prev) /(1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        self.decoder_actor = DiTActor(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['actor'])
        if expt == "expt_1":
            self.noisy_actions_to_critic = hp_dict['natc']
            self.q_ip = False
            self.decoder_critic1 = OGCritic(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'])
            self.decoder_critic2 = OGCritic(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'])
        elif expt == "expt_2":
            self.q_ip = True
            self.decoder_critic1 = DiTCritic(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'])
            self.decoder_critic2 = DiTCritic(hp_dict['state_dim'], hp_dict['model_dim'], self.action_dim, hp_dict['num_heads'], self.max_agents, hp_dict['dim_ff'], self.pos_embedding, hp_dict['dropout'], hp_dict['n_layers_dict']['critic'])

    def compute_log_prob_and_entropy(self, actions_k, actions_k_minus_1, mean, var, log_var):
        shape = actions_k.shape
        
        # Compute log probability of actions_k_minus_1 under this Gaussian distribution
        # For a multivariate Gaussian with diagonal covariance:
        # log p(x) = -0.5 * (log(2π) + log(det(Σ)) + (x-μ)ᵀΣ⁻¹(x-μ))
        mean_diff = actions_k_minus_1 - mean
        
        # Compute mahalanobis distance (x-μ)ᵀΣ⁻¹(x-μ)
        inv_variance = 1.0 / var
        mahalanobis = mean_diff.pow(2) * inv_variance
        
        # Sum over all dimensions except batch
        dims_to_sum = tuple(range(1, len(shape)))
        
        # Compute log prob: -0.5 * (n_dims * log(2π) + sum(log(σ²)) + sum((x-μ)²/σ²))
        n_dims = np.prod(shape[1:])  # Total dimensions per batch element
        log_prob = -0.5 * (n_dims * np.log(2 * np.pi) + 
                            log_var.sum(dim = -1) + 
                            mahalanobis.sum(dim = -1))
        
        # Compute entropy: For Gaussian with diagonal covariance: 
        # H = 0.5 * (n_dims * (1 + log(2π)) + sum(log(σ²)))
        entropy = 0.5 * (n_dims * (1.0 + np.log(2 * np.pi)) + 
                        log_var.sum(dim=dims_to_sum))
        
        return log_prob, entropy

    def sample_q(self, x_0, k, noise):
        return (_extract_into_tensor(self.sqrt_alphas_cumprod, k, x_0.shape) * x_0
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_0.shape) * noise)

    def get_actions(self, x_K, states, pos, get_low_level=True):
        # actions get denoised from x_K --> x_K --> x_0
        # With no_grad we can replace the actions in-place and return final denoised actions.
        actions = x_K
        with torch.no_grad():
            if get_low_level:
                log_ps = np.zeros((x_K.shape[0], self.diff_timesteps, x_K.shape[1]))
                a_ks = np.zeros((x_K.shape[0], self.diff_timesteps, *x_K.shape[1:]))
                ents = np.zeros((x_K.shape[0], self.diff_timesteps, 1))
                for k in reversed(range(self.diff_timesteps)):
                    actions, log_p, H, _ = self.get_actions_k_minus_1(k, actions, states, pos, get_low_level)
                    log_ps[:, k, :] = log_p.detach().cpu().numpy() # bs, N
                    a_ks[:, k, :, :] = actions.detach().cpu().numpy() # bs, N, act_dim
                    ents[:, k] = H.detach().cpu().numpy()[:, None] # bs, 1
                    
                actions = actions.detach().cpu().numpy()
                # return np.clip(actions.detach().cpu().numpy(), self.act_lim_low, self.act_lim_high), a_ks, log_ps, ents
            else:
                a_ks, log_ps, ents = None, None, None
                for k in reversed(range(self.diff_timesteps)):
                    actions = self.get_actions_k_minus_1(k, actions, states, pos, get_low_level)
                    
        return actions, a_ks, log_ps, ents
    
    def get_actions_k_minus_1(self, k, actions, states, pos, get_low_level=True):
        shape = actions.shape
            
        k = torch.tensor([k]*shape[0], device=self.device)
        pred_noise = self.decoder_actor(states, actions, pos)
        denoised_actions, mean, var, log_var = self.denoising_core(k, shape, actions, pred_noise)
        
        # Compute the noise from noisy actions as target for pred_noise
        sqrt_alphas_cumprod = _extract_into_tensor(self.sqrt_alphas_cumprod, k, shape)
        sqrt_one_minus_alphas_cumprod = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, shape)
        noise_target = (actions - sqrt_alphas_cumprod * denoised_actions) / sqrt_one_minus_alphas_cumprod
        
        mse_loss = F.mse_loss(pred_noise, noise_target, reduction='none')
        mse_loss = reduce(mse_loss, 'b ... -> b(...)', 'mean')
        
        if get_low_level:
            log_p, entropy = self.compute_log_prob_and_entropy(actions, denoised_actions, mean, var, log_var)
            return denoised_actions, log_p, entropy, mse_loss.mean()
        else:
            return denoised_actions, mse_loss
        
    def denoising_core(self, k, shape, inputs, pred_noise):
        model_variance = _extract_into_tensor(self.posterior_variance, k, shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, k, shape)
        
        pred_x_start = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, k, shape) * inputs\
                    - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, k, shape) * pred_noise
        
        model_mean = _extract_into_tensor(self.posterior_mean_coef1, k, shape) * pred_x_start\
                    + _extract_into_tensor(self.posterior_mean_coef2, k, shape) * inputs
        
        noise = torch.randn(shape, device=self.device)
        nonzero_mask = ((k != 0).float().view(-1, *([1] * (len(shape) - 1))))
        
        denoised_inputs = model_mean + nonzero_mask * model_variance * noise
        return denoised_inputs, model_mean, model_variance, model_log_variance
    
    def get_q_values(self, critic, x_T, states, actions, pos):
        q_values = x_T
        intermediate_q_values = []
        for k in reversed(range(self.diff_timesteps)):
            q_values = self.get_q_values_k_minus_1(critic, k, q_values, states, actions, pos)
            intermediate_q_values.append(q_values.detach().cpu().numpy())
        return q_values, np.array(intermediate_q_values)
        
    def get_q_values_k_minus_1(self, critic, k, q_values, states, actions, pos):
        shape = q_values.shape
            
        k = torch.tensor([k]*shape[0], device=self.device)
        ### p_mean_variance
        pred_noise = critic(states, actions, pos, q_values)

        model_variance = _extract_into_tensor(self.posterior_variance, k, shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, k, shape)

        pred_x_start = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, k, shape) * q_values\
                    - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, k, shape) * pred_noise
        
        model_mean = _extract_into_tensor(self.posterior_mean_coef1, k, shape) * pred_x_start\
                    + _extract_into_tensor(self.posterior_mean_coef2, k, shape) * q_values
        
        ### p_sample
        noise = torch.randn(shape, device=self.device)
        nonzero_mask = ((k != 0).float().view(-1, *([1] * (len(shape) - 1))))
        q_values = model_mean + nonzero_mask * model_variance * noise
        return q_values
    
    def compute_pi_loss(self, actions, states, pos, ent_ret=False):
        shape = actions.shape
        bs, n_agents, _ = states.size()
        noise = torch.randn(shape, device=self.device)
        ks = torch.randint(0, self.diff_timesteps, (bs,), device=self.device)

        # denoising transition mean calculation
        noisy_actions = self.sample_q(actions, ks, noise)
        pred_noise = self.decoder_actor(states, noisy_actions, pos)
        
        #** For Entropy Experiment. Can delete later. ###################################################
        if ent_ret:
            denoised_actions, mean, var, log_var = self.denoising_core(ks, shape, actions, pred_noise)
            log_p, entropy = self.compute_log_prob_and_entropy(actions, denoised_actions, mean, var, log_var)
        else:
            entropy = None
        #**##############################################################################################
        
        loss = F.mse_loss(noise, pred_noise, reduction='none')
        loss = reduce(loss, 'b ... -> b(...)', 'mean')
        loss = loss.mean()
        return loss, entropy, ks
    
    def compute_q_loss(self, critic, states, actions, pos, rewards):
        bs, n_agents, _ = states.size()
        t_steps = torch.randint(0, self.diff_timesteps, (bs,), device=self.device)
        rewards = rewards.unsqueeze(1).expand(-1, n_agents).unsqueeze(-1)

        if self.q_ip:
            noise = torch.randn(rewards.shape, device=self.device)
            noisy_actions = self.sample_q(actions, t_steps, noise)
            
            noisy_q = self.sample_q(rewards, t_steps, noise)
            pred_noise = critic(states, noisy_actions, pos, noisy_q)
            loss = F.mse_loss(noise, pred_noise)
        else:
            # Here all actions noisy and unnoisy predict the same final reward
            if self.noisy_actions_to_critic:
                noise = torch.randn(actions.shape, device=self.device)
                noisy_actions = self.sample_q(actions, t_steps, noise)
                pred_noise = critic(states, noisy_actions, pos)
                loss = F.mse_loss(rewards, pred_noise)
            else:
                q = critic(states, actions, pos).squeeze() #.mean(dim=1)
                q_next = rewards.unsqueeze(-1) #+ self.hp_dict['gamma'] * ((1 - d.unsqueeze(1)) * (next_q - self.alpha * log_probs)).mean(dim=1)
                loss = F.mse_loss(q, q_next)
        
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

