"""Configuration for SSL semantic communication training."""

from dataclasses import dataclass
import torch


@dataclass
class StageAConfig:
    """Configuration for Stage A (SSL semantic pretraining)."""
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Representation + channel
    z_dim: int = 256
    sym_T: int = 64  # channel symbol length (I/Q packed => 2*T real dims)

    # SSL hyperparameters
    tau: float = 0.2  # temperature for InfoNCE loss
    lambda_rob: float = 0.1  # weight for robustness loss

    # Channel parameters
    snr_db_min: float = 0.0
    snr_db_max: float = 20.0

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Training
    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 4
    log_every: int = 100  # steps
