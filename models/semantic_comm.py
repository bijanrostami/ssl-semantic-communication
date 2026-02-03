"""Semantic communication modules (modulator, receiver, etc.)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modulator(nn.Module):
    """
    Map semantic vector to real-valued I/Q-packed symbols.
    
    Args:
        z_dim: Dimension of semantic embedding
        sym_T: Number of complex symbols (output is 2*sym_T real values)
    """
    
    def __init__(self, z_dim: int, sym_T: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * sym_T),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Semantic embeddings [B, z_dim]
            
        Returns:
            Power-normalized symbols [B, 2*sym_T]
        """
        s = self.net(z)
        # Power normalize per sample
        s = s / (s.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
        return s


class Receiver(nn.Module):
    """
    Map received symbols back to semantic embedding.
    
    Args:
        sym_T: Number of complex symbols (input is 2*sym_T real values)
        z_dim: Dimension of semantic embedding
    """
    
    def __init__(self, sym_T: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * sym_T, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, z_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Received symbols [B, 2*sym_T]
            
        Returns:
            Reconstructed semantic embeddings [B, z_dim]
        """
        return self.net(y)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive loss (SimCLR-style).
    
    Args:
        z_dim: Dimension of input embeddings
        proj_dim: Dimension of projected embeddings
    """
    
    def __init__(self, z_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Semantic embeddings [B, z_dim]
            
        Returns:
            L2-normalized projections [B, proj_dim]
        """
        u = self.net(z)
        return F.normalize(u, dim=-1)


class MaskPredictor(nn.Module):
    """
    Predict masked semantic dimensions (for masked prediction loss).
    
    Args:
        z_dim: Dimension of semantic embeddings
    """
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Masked semantic embeddings [B, z_dim]
            
        Returns:
            Predicted values for all dimensions [B, z_dim]
        """
        return self.net(z)
