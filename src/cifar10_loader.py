"""CIFAR-10 Dataset Loader for MAE Pretraining

This module provides a PyTorch DataLoader for CIFAR-10 that:
- Resizes images to 224x224 (standard ViT/MAE input size)
- Applies ImageNet normalization statistics
- Optimizes for GPU training with pin_memory=True
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=128, num_workers=4, data_root='./data'):
    """
    Create CIFAR-10 DataLoaders with MAE pretraining configuration.
    
    Args:
        batch_size (int): Batch size for training/testing. Default: 128
        num_workers (int): Number of worker processes for data loading. Default: 4
        data_root (str): Root directory to store CIFAR-10 data. Default: './data'
    
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    
    # ImageNet normalization statistics for compatibility with MAE
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Training transforms: resize to 224x224 and normalize
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Test transforms: same as training for consistency
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create DataLoaders optimized for GPU training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Useful for batch norm and consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == '__main__':
    # Example usage
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_loaders(
        batch_size=128,
        num_workers=4
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    
    # Verify data loading and normalization
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'\nBatch {batch_idx}:')
        print(f'  Images shape: {images.shape}')
        print(f'  Labels shape: {labels.shape}')
        print(f'  Image value range: [{images.min():.3f}, {images.max():.3f}]')
        print(f'  Mean per channel: {images.mean([0, 2, 3])}')
        print(f'  Std per channel: {images.std([0, 2, 3])}')
        break  # Only show first batch
