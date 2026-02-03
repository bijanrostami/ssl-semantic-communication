"""Wireless channel simulation."""

from typing import Tuple
import torch


def sample_snr_db(snr_min: float, snr_max: float, batch_size: int, device: str) -> torch.Tensor:
    """
    Sample random SNR values uniformly in dB scale.
    
    Args:
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        batch_size: Number of samples
        device: Device to place tensor on
        
    Returns:
        SNR values in dB [batch_size]
    """
    return snr_min + (snr_max - snr_min) * torch.rand(batch_size, device=device)


def rayleigh_awgn_channel(s: torch.Tensor, snr_db: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Real scalar Rayleigh fading + AWGN channel.
    
    Args:
        s: Transmitted symbols [B, 2T]
        snr_db: SNR in dB for each sample [B]
        
    Returns:
        y: Received symbols [B, 2T]
        h: Channel coefficients [B, 1]
    """
    B, _ = s.shape
    device = s.device

    # Rayleigh fading coefficient
    h = torch.sqrt(
        0.5 * (torch.randn(B, 1, device=device) ** 2 + torch.randn(B, 1, device=device) ** 2) + 1e-8
    )

    # Calculate noise power based on signal power and SNR
    sig_pow = s.pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    snr_lin = 10 ** (snr_db.view(B, 1) / 10.0)
    noise_pow = (h.pow(2) * sig_pow) / (snr_lin + 1e-8)
    
    # Generate AWGN
    noise = torch.randn_like(s) * torch.sqrt(noise_pow + 1e-8)

    # Apply channel
    y = h * s + noise
    return y, h
