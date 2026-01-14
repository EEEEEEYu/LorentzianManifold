"""
Loss functions for Causal Autoencoder
Based on the research proposal
"""

import torch
import torch.nn.functional as F


def causal_ordering_loss(mu, margin=0.1, K=3):
    """
    Enforce that frame t is in past light cone of frame t+k.
    
    Args:
        mu: [batch, T, 129]
        margin: Hinge loss margin
        K: Temporal window (check up to K frames ahead)
    """
    batch, T, _ = mu.shape
    loss = 0.0
    count = 0
    
    for k in range(1, min(K+1, T)):
        mu_current = mu[:, :-k, :]  # [batch, T-k, 129]
        mu_future = mu[:, k:, :]     # [batch, T-k, 129]
        
        # Compute Δs²
        tau_curr = mu_current[..., 0]
        tau_future = mu_future[..., 0]
        x_curr = mu_current[..., 1:]
        x_future = mu_future[..., 1:]
        
        delta_tau = tau_future - tau_curr  # Should be positive
        delta_x = x_future - x_curr
        ds2 = -delta_tau**2 + torch.sum(delta_x**2, dim=-1)
        
        # Should be time-like (ds² < 0)
        loss_k = F.relu(ds2 + margin).mean()
        loss += loss_k
        count += 1
    
    return loss / count if count > 0 else torch.tensor(0.0, device=mu.device)


def spatial_independence_loss(mu, margin=0.1, num_samples=100):
    """
    Sample random frame pairs that are far apart in time.
    They should have space-like separation (independent).
    
    Args:
        mu: [batch, T, 129]
        margin: Hinge loss margin
        num_samples: Number of random pairs to sample
    """
    batch, T, _ = mu.shape
    
    if T < 10:  # Not enough temporal separation
        return torch.tensor(0.0, device=mu.device)
    
    loss = 0.0
    
    for _ in range(num_samples):
        # Sample two frames far apart (at least T//3 frames apart)
        i = torch.randint(0, T - T//3, (batch,), device=mu.device)
        # j must be >= i + T//3
        # For each batch item, sample j from [i + T//3, T)
        j_list = []
        for b in range(batch):
            j_val = torch.randint(i[b].item() + T//3, T, (1,), device=mu.device).item()
            j_list.append(j_val)
        j = torch.tensor(j_list, device=mu.device, dtype=torch.long)
        
        mu_i = mu[torch.arange(batch, device=mu.device), i]  # [batch, 129]
        mu_j = mu[torch.arange(batch, device=mu.device), j]  # [batch, 129]
        
        tau_i, x_i = mu_i[:, 0], mu_i[:, 1:]
        tau_j, x_j = mu_j[:, 0], mu_j[:, 1:]
        
        delta_tau = tau_j - tau_i
        delta_x = x_j - x_i
        ds2 = -delta_tau**2 + torch.sum(delta_x**2, dim=-1)
        
        # Should be space-like (ds² > 0)
        loss += F.relu(-ds2 + margin).mean()
    
    return loss / num_samples


def temporal_ordering_loss(mu, epsilon=0.01):
    """
    Ensure τ_{t+1} > τ_t (monotonicity).
    """
    tau = mu[..., 0]  # [batch, T]
    tau_diff = tau[:, 1:] - tau[:, :-1]  # [batch, T-1]
    
    # Penalize if τ doesn't increase
    loss = F.relu(-tau_diff + epsilon).mean()
    return loss


def hyperboloid_consistency_loss(mu, c=1.0):
    """
    Soft constraint: -τ² + ||x||² ≈ -c²
    (Backup to hard projection in forward pass)
    """
    tau = mu[..., 0]  # [batch, T]
    x = mu[..., 1:]    # [batch, T, 128]
    
    x_norm_sq = torch.sum(x**2, dim=-1)  # [batch, T]
    constraint = -tau**2 + x_norm_sq + c**2
    
    loss = (constraint ** 2).mean()
    return loss


def compute_causal_loss(frames_input, frames_recon, mu, loss_weights=None):
    """
    Compute total loss for causal autoencoder.
    
    Args:
        frames_input: [batch, T, 3, H, W] original frames
        frames_recon: [batch, T, 3, H, W] reconstructed frames
        mu: [batch, T, 129] Lorentzian embeddings
        loss_weights: Dict with weights for each loss component
    Returns:
        loss: Total loss
        loss_dict: Dict with individual loss components
    """
    if loss_weights is None:
        loss_weights = {
            'recon': 1.0,
            'causal': 1.0,
            'acausal': 0.5,
            'temporal': 0.1,
            'hyperboloid': 0.01
        }
    
    # Loss 1: Reconstruction (semantic preservation)
    L_recon = F.mse_loss(frames_recon, frames_input)
    
    # Loss 2: Temporal causality (enforce time-like separation)
    L_causal = causal_ordering_loss(mu)
    
    # Loss 3: Spatial acausality (enforce space-like separation)
    L_acausal = spatial_independence_loss(mu)
    
    # Loss 4: Temporal monotonicity
    L_temporal = temporal_ordering_loss(mu)
    
    # Loss 5: Hyperboloid constraint (soft, backup to hard projection)
    L_hyperboloid = hyperboloid_consistency_loss(mu)
    
    # Total
    loss = (loss_weights['recon'] * L_recon 
            + loss_weights['causal'] * L_causal 
            + loss_weights['acausal'] * L_acausal 
            + loss_weights['temporal'] * L_temporal 
            + loss_weights['hyperboloid'] * L_hyperboloid)
    
    return loss, {
        'recon': L_recon.item(),
        'causal': L_causal.item(),
        'acausal': L_acausal.item(),
        'temporal': L_temporal.item(),
        'hyperboloid': L_hyperboloid.item()
    }
