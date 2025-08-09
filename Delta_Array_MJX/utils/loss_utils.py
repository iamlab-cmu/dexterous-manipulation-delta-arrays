import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def matsac_q_loss(tf, tf_target, gamma, alpha, s1, a, s2, r, d, pos, act_dim):
    """
    Q-function is NOT a Diffusion model.
    Q-value is evaluated at each denoisified action step within the low-level MDP.
    """
    q1 = tf.decoder_critic1(s1, a, pos).squeeze().mean(dim=1)
    q2 = tf.decoder_critic2(s1, a, pos).squeeze().mean(dim=1)
    
    with torch.no_grad():
        # Get 1 step denoised action from policy
        bs, N, _ = s2.size()
        a_K = torch.randn((bs, N, act_dim), device=self.device)
        next_actions = tf.get_actions(a, s2, pos)
        
        next_q1 = tf_target.decoder_critic1(s2, next_actions, pos).squeeze()
        next_q2 = tf_target.decoder_critic2(s2, next_actions, pos).squeeze()
        q_next = r + gamma * ((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2))).mean(dim=1)

    q_loss1 = F.mse_loss(q1, q_next)
    q_loss2 = F.mse_loss(q2, q_next)
    return q_loss1, q_loss2

def matsac_q_loss_diff(tf, tf_target, gamma, alpha, k, s1, a, s2, r, d, pos):
    """
    Q-function IS a Diffusion model.
    Q-value is evaluated at each denoisified action step within the low-level MDP.
    """
    q1 = tf.decoder_critic1(k, s1, a, pos).squeeze()
    q2 = tf.decoder_critic2(k, s1, a, pos).squeeze()
    
    with torch.no_grad():    
        next_actions = tf.get_actions_k_minus_1(s2, pos)
        
        next_q1 = tf_target.decoder_critic1(k, s2, next_actions, pos).squeeze()
        next_q2 = tf_target.decoder_critic2(k, s2, next_actions, pos).squeeze()
        q_next = r.unsqueeze(1) + gamma * ((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2)))

    q_loss1 = F.mse_loss(q1, q_next)
    q_loss2 = F.mse_loss(q2, q_next)
    return q_loss1, q_loss2