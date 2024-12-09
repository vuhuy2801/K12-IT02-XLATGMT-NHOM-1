from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data pipeline"""
    data_dir: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.75
    val_split: float = 0.15
    test_split: float = 0.10
    pin_memory: bool = True

class AnimalDataModule:
    """Handles data loading and preprocessing for the Animals-90 dataset"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
        self.class_to_idx: Optional[Dict[str, int]] = None
        
        # Validate data directory
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
            
    def _get_train_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size), 
                            interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=15,
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _get_val_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size),
                            interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def setup(self) -> Tuple[Dataset, Dataset, Dataset]:
        try:
            full_dataset = datasets.ImageFolder(
                root=str(self.config.data_dir),
                transform=None
            )
            
            self.class_to_idx = full_dataset.class_to_idx
            logger.info(f"Found {len(full_dataset)} images in {len(self.class_to_idx)} classes")
            
            total_size = len(full_dataset)
            train_size = int(self.config.train_split * total_size)
            val_size = int(self.config.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_dataset.dataset.transform = self.train_transforms
            val_dataset.dataset.transform = self.val_transforms
            test_dataset.dataset.transform = self.val_transforms
            
            logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error setting up datasets: {str(e)}")
            raise
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = self.setup()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, test_loader

    def get_class_weights(self) -> torch.Tensor:
        if not self.class_to_idx:
            raise RuntimeError("Dataset not setup yet. Call setup() first")
            
        try:
            class_counts = np.zeros(len(self.class_to_idx))
            train_dataset, _, _ = self.setup()
            
            for _, label in train_dataset:
                class_counts[label] += 1
                
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * len(self.class_to_idx)
            
            return torch.from_numpy(weights.astype(np.float32))
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {str(e)}")
            raise

if __name__ == "__main__":
    config = DataConfig(
        data_dir=Path("dataset"),
        image_size=224,
        batch_size=32,
        num_workers=4
    )
    
    data_module = AnimalDataModule(config)
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    class_weights = data_module.get_class_weights()
    
    # Print sample batch
    images, labels = next(iter(train_loader))
    logger.info(f"Sample batch shape: {images.shape}")
    logger.info(f"Sample labels shape: {labels.shape}") 