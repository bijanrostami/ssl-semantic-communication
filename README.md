# SSL Semantic Communication

Self-supervised learning (SSL) framework for semantic communication over wireless channels using CIFAR-10.

## Overview

This project implements Stage A of a semantic communication system that uses self-supervised learning to learn robust representations for transmission over noisy wireless channels. The system consists of:

- **Encoder**: CNN encoder that extracts semantic features from CIFAR-10 images
- **Modulator**: Maps semantic embeddings to channel symbols
- **Channel**: Rayleigh fading + AWGN wireless channel simulation
- **Receiver**: Reconstructs semantic embeddings from received symbols
- **SSL Training**: Uses contrastive learning (InfoNCE) and robustness regularization

## Project Structure

```
ssl-semantic-communication/
├── models/
│   ├── __init__.py         # Model exports
│   ├── encoder.py          # CNN encoder for CIFAR-10
│   ├── channel.py          # Wireless channel simulation
│   └── semantic_comm.py    # Modulator, receiver, projection head
├── config.py               # Configuration dataclass
├── dataset.py              # CIFAR-10 data loading and augmentation
├── losses.py               # Loss functions (InfoNCE, robustness)
├── model.py                # Stage A model wrapper
├── train.py                # Training script
├── eval.py                 # Evaluation (linear probe)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bijanrostami/ssl-semantic-communication.git
cd ssl-semantic-communication
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the Stage A model with default configuration:

```bash
python train.py
```

The training script will:
- Download CIFAR-10 automatically if not present
- Train for 200 epochs with augmented views
- Evaluate with linear probe every 20 epochs
- Save checkpoints to `checkpoints/` directory

### Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py checkpoints/stage_a_final.pt
```

This trains a linear classifier on frozen embeddings and reports test accuracy.

## Configuration

Edit [config.py](config.py) to modify hyperparameters:

```python
@dataclass
class StageAConfig:
    # Model architecture
    z_dim: int = 256          # Semantic embedding dimension
    sym_T: int = 64           # Channel symbol length
    
    # SSL hyperparameters
    tau: float = 0.2          # InfoNCE temperature
    lambda_rob: float = 0.1   # Robustness loss weight
    
    # Channel parameters
    snr_db_min: float = 0.0   # Minimum SNR in dB
    snr_db_max: float = 20.0  # Maximum SNR in dB
    
    # Training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
```

## Method

### Self-Supervised Learning

The model uses a SimCLR-inspired approach:
1. Generate two augmented views of each image
2. Pass through encoder → modulator → channel → receiver
3. Apply contrastive loss (InfoNCE) on projected embeddings
4. Add robustness regularization (MSE between embeddings)

### Channel Simulation

Realistic wireless channel with:
- Rayleigh fading (time-varying channel coefficients)
- Additive white Gaussian noise (AWGN)
- Random SNR sampled from configured range

### Key Features

- **End-to-end differentiable**: Channel is differentiable for gradient-based learning
- **SNR-robust**: Trains with variable SNR to learn robust representations
- **Modular design**: Easy to extend with new encoders, channels, or loss functions

## Results

Linear probe accuracy on CIFAR-10 test set (after 200 epochs):
- **@10 dB SNR**: ~XX% (update with your results)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ssl_semantic_comm,
  author = {Your Name},
  title = {SSL Semantic Communication},
  year = {2026},
  url = {https://github.com/bijanrostami/ssl-semantic-communication}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com
