"""Stage A SSL model wrapper."""

import torch
import torch.nn as nn
from typing import Tuple

from config import StageAConfig
from models import CIFAREncoder, Modulator, Receiver, ProjectionHead
from models.channel import rayleigh_awgn_channel, sample_snr_db
from losses import info_nce_loss, robustness_loss


class StageAModel(nn.Module):
    """
    Complete Stage A model for SSL semantic communication.
    
    Architecture:
        Encoder -> Modulator -> Channel -> Receiver -> Projection Head
        
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg: StageAConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = CIFAREncoder(z_dim=cfg.z_dim)
        self.mod = Modulator(cfg.z_dim, cfg.sym_T)
        self.rx = Receiver(cfg.sym_T, cfg.z_dim)
        self.proj = ProjectionHead(cfg.z_dim, proj_dim=128)

    def forward_one_view(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one augmented view.
        
        Args:
            x: Input images [B, 3, 32, 32]
            snr_db: SNR in dB for each sample [B]
            
        Returns:
            Received embeddings [B, z_dim]
        """
        z = self.encoder(x)
        s = self.mod(z)
        y, _ = rayleigh_awgn_channel(s, snr_db)
        z_hat = self.rx(y)
        return z_hat

    @torch.no_grad()
    def extract_embeddings(self, x: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """
        Extract embeddings for evaluation (frozen, no gradients).
        
        Args:
            x: Input images [B, 3, 32, 32]
            snr_db: Fixed SNR in dB for evaluation
            
        Returns:
            Embeddings [B, z_dim]
        """
        self.eval()
        x = x.to(self.cfg.device, non_blocking=True)
        B = x.size(0)
        snr = torch.full((B,), float(snr_db), device=self.cfg.device)
        z_hat = self.forward_one_view(x, snr_db=snr)
        return z_hat


def stage_a_step(model: StageAModel, x1: torch.Tensor, x2: torch.Tensor, 
                 cfg: StageAConfig) -> Tuple[torch.Tensor, dict]:
    """
    Single training step for Stage A.
    
    Args:
        model: StageAModel instance
        x1: First augmented view [B, 3, 32, 32]
        x2: Second augmented view [B, 3, 32, 32]
        cfg: Configuration object
        
    Returns:
        loss: Total loss (scalar)
        stats: Dictionary of loss components for logging
    """
    model.train()
    x1 = x1.to(cfg.device, non_blocking=True)
    x2 = x2.to(cfg.device, non_blocking=True)
    B = x1.size(0)

    # Sample different SNR for each view
    snr1 = sample_snr_db(cfg.snr_db_min, cfg.snr_db_max, B, cfg.device)
    snr2 = sample_snr_db(cfg.snr_db_min, cfg.snr_db_max, B, cfg.device)

    # Forward both views
    zhat1 = model.forward_one_view(x1, snr1)
    zhat2 = model.forward_one_view(x2, snr2)

    # Contrastive loss
    u1 = model.proj(zhat1)
    u2 = model.proj(zhat2)
    loss_nce = info_nce_loss(u1, u2, cfg.tau)

    # Robustness regularization
    loss_rob = robustness_loss(zhat1, zhat2)

    # Total loss
    loss = loss_nce + cfg.lambda_rob * loss_rob

    stats = {
        "loss": float(loss.detach().cpu()),
        "nce": float(loss_nce.detach().cpu()),
        "rob": float(loss_rob.detach().cpu()),
    }
    
    return loss, stats
