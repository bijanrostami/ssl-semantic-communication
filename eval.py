"""Linear probe evaluation for SSL representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import StageAConfig
from dataset import build_cifar10_eval_loaders
from model import StageAModel


class LinearProbe(nn.Module):
    """Linear classifier head for evaluation."""
    
    def __init__(self, z_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(z_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def linear_probe_eval(
    model: StageAModel,
    cfg: StageAConfig,
    snr_db_eval: float = 10.0,
    probe_epochs: int = 10,
    probe_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> float:
    """
    Train a linear classifier on frozen embeddings and return test accuracy.
    
    Args:
        model: Trained Stage A model
        cfg: Configuration object
        snr_db_eval: Fixed SNR for evaluation in dB
        probe_epochs: Number of epochs to train the linear probe
        probe_lr: Learning rate for probe training
        weight_decay: Weight decay for probe training
        
    Returns:
        Test accuracy (%)
    """
    # Freeze Stage A model
    model.eval()
    
    # Build evaluation loaders
    train_loader, test_loader = build_cifar10_eval_loaders(
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers
    )
    
    # Initialize linear probe
    probe = LinearProbe(cfg.z_dim, num_classes=10).to(cfg.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=weight_decay)
    
    # Train probe
    for ep in range(probe_epochs):
        probe.train()
        correct, total, loss_sum = 0, 0, 0.0
        
        for x, y in train_loader:
            y = y.to(cfg.device, non_blocking=True)
            
            # Extract frozen embeddings
            with torch.no_grad():
                z = model.extract_embeddings(x, snr_db=snr_db_eval)
            
            # Train classifier
            logits = probe(z.detach())
            loss = F.cross_entropy(logits, y)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            # Metrics
            loss_sum += float(loss.detach().cpu()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach().cpu())
            total += y.size(0)
        
        train_acc = 100.0 * correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        
        # Evaluate on test set
        probe.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                y = y.to(cfg.device, non_blocking=True)
                z = model.extract_embeddings(x, snr_db=snr_db_eval)
                logits = probe(z)
                pred = logits.argmax(dim=1)
                correct_t += int((pred == y).sum().detach().cpu())
                total_t += y.size(0)
        
        test_acc = 100.0 * correct_t / max(1, total_t)
        print(f"[LinearProbe | ep {ep+1:02d}/{probe_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"test_acc={test_acc:.2f}% (SNR={snr_db_eval} dB)")
    
    return test_acc


if __name__ == "__main__":
    """Example: Evaluate a trained checkpoint."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval.py <checkpoint_path>")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint_path = sys.argv[1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Restore config and model
    cfg = StageAConfig(**checkpoint["cfg"])
    model = StageAModel(cfg).to(cfg.device)
    model.load_state_dict(checkpoint["model"])
    
    # Run evaluation
    print(f"Evaluating checkpoint: {checkpoint_path}")
    acc = linear_probe_eval(model, cfg, snr_db_eval=10.0, probe_epochs=10)
    print(f"\nFinal test accuracy @10 dB: {acc:.2f}%")
