"""CNN encoder for CIFAR-10 images."""

import torch
import torch.nn as nn


class CIFAREncoder(nn.Module):
    """
    CNN encoder for CIFAR-10 (3x32x32) -> z_dim.
    
    Args:
        z_dim: Dimension of the output semantic embedding
    """
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, 32, 32]
            
        Returns:
            Semantic embeddings [B, z_dim]
        """
        h = self.features(x).flatten(1)
        z = self.fc(h)
        return z
