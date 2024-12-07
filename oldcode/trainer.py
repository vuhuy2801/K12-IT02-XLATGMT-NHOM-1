from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import time
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import AnimalClassifier, ModelConfig
from data import DataConfig, AnimalDataModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    num_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    checkpoint_dir: Path = Path("checkpoints")
    tensorboard_dir: Path = Path("runs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision training

class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

class Trainer:
    """Training pipeline for animal classification model"""
    
    def __init__(
        self,
        model: AnimalClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set up device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Initialize monitoring
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(self.val_loader, desc="Validation"):
            images, targets = images.to(self.device), targets.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self) -> Dict[str, List[float]]:
        """Run complete training pipeline"""
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        try:
            for epoch in range(self.config.num_epochs):
                logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                
                # Training phase
                train_metrics = self.train_epoch()
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                
                # Validation phase
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Log metrics
                metrics = {
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy']
                }
                
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, epoch)
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, metrics)
                
                # Early stopping check
                if self.early_stopping(val_metrics['loss']):
                    logger.info("Early stopping triggered")
                    break
                
                logger.info(
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Train Acc: {train_metrics['accuracy']:.2f}% "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.writer.close()
        
        return history

def train_model(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: Optional[TrainingConfig] = None
) -> Tuple[AnimalClassifier, Dict[str, List[float]]]:
    """Complete training pipeline function"""
    if train_config is None:
        train_config = TrainingConfig()
    
    # Set up data
    data_module = AnimalDataModule(data_config)
    train_loader, val_loader, _ = data_module.get_data_loaders()
    
    # Create model
    model = AnimalClassifier(model_config)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, train_config)
    
    # Train model
    history = trainer.train()
    
    return model, history

if __name__ == "__main__":
    # Configurations
    data_config = DataConfig(
        data_dir=Path("dataset"),
        batch_size=128
    )
    
    model_config = ModelConfig(
        num_classes=90,
        pretrained=True
    )
    
    train_config = TrainingConfig(
        num_epochs=50,
        learning_rate=0.001,
        checkpoint_dir=Path("checkpoints"),
        tensorboard_dir=Path("runs/animal_classifier_" + 
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    
    # Train model
    model, history = train_model(data_config, model_config, train_config)