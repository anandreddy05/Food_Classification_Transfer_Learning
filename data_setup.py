from typing import Callable, Tuple, List
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from utils import train_transform,test_transform

def create_dataloader(
    train_path: str,
    test_path: str,
    train_transform: Callable,
    test_transform: Callable,
    batch_size: int = 32,
    num_workers: int = os.cpu_count() or 1,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates training and testing DataLoaders.
    
    Args:
        train_path: Path to training data folder.
        test_path: Path to testing data folder.
        train_transform: Function returning transforms for training data.
        test_transform: Function returning transforms for testing data.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory (useful for GPU training).
    
    Returns:
        Tuple of (train_dataloader, test_dataloader, class_names)
    """
    # Get transforms
    train_transforms = train_transform()
    test_transforms = test_transform()
    
    # Load datasets
    train_data = ImageFolder(root=train_path, transform=train_transforms)
    test_data = ImageFolder(root=test_path, transform=test_transforms)
    
    class_names = train_data.classes
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, test_dataloader, class_names
