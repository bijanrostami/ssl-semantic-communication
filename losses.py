"""Loss functions for SSL semantic communication."""

import torch
import torch.nn.functional as F


def info_nce_loss(u1: torch.Tensor, u2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss (SimCLR-style).
    
    Args:
        u1: Projected embeddings from view 1 [B, proj_dim]
        u2: Projected embeddings from view 2 [B, proj_dim]
        tau: Temperature parameter
        
    Returns:
        InfoNCE loss (scalar)
    """
    B = u1.size(0)
    logits = (u1 @ u2.t()) / tau
    labels = torch.arange(B, device=u1.device)
    
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    
    return 0.5 * (loss_12 + loss_21)


def robustness_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss between embeddings from two views (robustness regularization).
    
    Args:
        z1: Embeddings from view 1 [B, z_dim]
        z2: Embeddings from view 2 [B, z_dim]
        
    Returns:
        MSE loss (scalar)
    """
    return F.mse_loss(z1, z2)
