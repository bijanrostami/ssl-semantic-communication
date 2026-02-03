"""Training script for Stage A SSL semantic communication."""

import torch
import torch.nn as nn

from config import StageAConfig
from dataset import build_cifar10_ssl_loader
from model import StageAModel, stage_a_step
from eval import linear_probe_eval


def train_stage_a(cfg: StageAConfig):
    """
    Main training loop for Stage A.
    
    Args:
        cfg: Configuration object
    """
    # Build data loader
    train_loader = build_cifar10_ssl_loader(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    
    # Initialize model and optimizer
    model = StageAModel(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    
    print(f"Starting Stage A training on {cfg.device}")
    print(f"Total epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
    print(f"SNR range: {cfg.snr_db_min}-{cfg.snr_db_max} dB")
    print("-" * 60)
    
    global_step = 0
    
    for epoch in range(cfg.epochs):
        for (x1, x2), _ in train_loader:
            # Training step
            loss, stats = stage_a_step(model, x1, x2, cfg)
            
            # Optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            # Logging
            if global_step % cfg.log_every == 0:
                print(
                    f"[Epoch {epoch:03d} | Step {global_step:06d}] "
                    f"loss={stats['loss']:.4f} "
                    f"nce={stats['nce']:.4f} "
                    f"rob={stats['rob']:.4f}"
                )
            
            global_step += 1
        
        # Periodic evaluation and checkpointing
        if (epoch + 1) % 20 == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at epoch {epoch + 1}...")
            acc = linear_probe_eval(model, cfg, snr_db_eval=10.0, probe_epochs=10)
            print(f"Linear probe test accuracy @10 dB: {acc:.2f}%")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg.__dict__,
            }
            torch.save(checkpoint, f"checkpoints/stage_a_epoch{epoch+1}.pt")
            print(f"Saved checkpoint: checkpoints/stage_a_epoch{epoch+1}.pt\n")
    
    # Save final model
    final_checkpoint = {
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
    }
    torch.save(final_checkpoint, "checkpoints/stage_a_final.pt")
    print(f"\n{'='*60}")
    print("Training completed!")
    print("Saved final model: checkpoints/stage_a_final.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Initialize configuration
    cfg = StageAConfig()
    
    # Create checkpoints directory
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start training
    train_stage_a(cfg)
