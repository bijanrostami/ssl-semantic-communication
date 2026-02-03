"""CIFAR-10 dataset loading and augmentation transforms."""

import torch
from torchvision import datasets, transforms


class TwoCropsTransform:
    """Create two augmented views from the same PIL image (for SSL)."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        """
        Args:
            x: PIL image
            
        Returns:
            Tuple of two augmented views
        """
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        return x1, x2


def build_ssl_transform() -> TwoCropsTransform:
    """
    Build SimCLR-style augmentation for CIFAR-10 SSL training.
    
    Returns:
        Transform that produces two augmented views
    """
    base = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])
    return TwoCropsTransform(base)


def build_eval_transform():
    """
    Build simple transform for evaluation (no augmentation).
    
    Returns:
        Transform for evaluation
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def build_cifar10_ssl_loader(batch_size: int = 256, num_workers: int = 4, 
                              data_root: str = "./data"):
    """
    Build CIFAR-10 DataLoader for SSL training.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_root: Root directory for CIFAR-10 data
        
    Returns:
        DataLoader for SSL training
    """
    ssl_transform = build_ssl_transform()
    
    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=ssl_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader


def build_cifar10_eval_loaders(batch_size: int = 256, num_workers: int = 4,
                                data_root: str = "./data"):
    """
    Build CIFAR-10 DataLoaders for evaluation (linear probe).
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_root: Root directory for CIFAR-10 data
        
    Returns:
        Tuple of (train_loader, test_loader) for evaluation
    """
    eval_transform = build_eval_transform()
    
    train_set = datasets.CIFAR10(
        root=data_root, 
        train=True, 
        download=True, 
        transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root=data_root, 
        train=False, 
        download=True, 
        transform=eval_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=False
    )
    
    return train_loader, test_loader
