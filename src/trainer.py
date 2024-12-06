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

from .config import ModelConfig, DataConfig, TrainingConfig
from .model import AnimalValidationModel, AnimalClassifier, TwoStageClassifier
from .data import AnimalDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Xử lý early stopping cho training"""
    
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

class StageTrainer:
    """Base trainer class cho mỗi stage"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        stage_name: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stage_name = stage_name
        
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Khởi tạo components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1,
            patience=3, verbose=True
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Monitoring
        self.writer = SummaryWriter(
            config.tensorboard_dir / f"{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Tạo thư mục checkpoints
        self.checkpoint_dir = config.checkpoint_dir / stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Khởi tạo trainer cho {stage_name} với device: {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train một epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training {self.stage_name}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Chạy validation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in tqdm(self.val_loader, desc=f"Validating {self.stage_name}"):
            images, targets = images.to(self.device), targets.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Đã lưu checkpoint ti {path}")

class TwoStageTrainer:
    """Trainer chính quản lý việc huấn luyện cả 2 stage"""
    
    def __init__(
        self,
        model: TwoStageClassifier,
        data_module: AnimalDataModule,
        config: TrainingConfig
    ):
        self.model = model
        self.data_module = data_module
        self.config = config
        
        # Tạo trainers cho từng stage
        train_loader, val_loader, _ = data_module.get_data_loaders()
        
        self.stage1_trainer = StageTrainer(
            model.validation_model,
            train_loader,
            val_loader,
            config,
            "stage1_validation"
        )
        
        self.stage2_trainer = StageTrainer(
            model.classification_model,
            train_loader,
            val_loader,
            config,
            "stage2_classifier"
        )
    
    def train(self) -> Dict[str, Dict[str, List[float]]]:
        """Huấn luyện toàn bộ hệ thống"""
        history = {
            'stage1': {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            },
            'stage2': {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            }
        }
        
        # Train Stage 1: Validation model
        logger.info("=== Bắt đầu huấn luyện Stage 1: Validation Model ===")
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            train_metrics = self.stage1_trainer.train_epoch()
            val_metrics = self.stage1_trainer.validate()
            
            # Cập nhật history
            for k, v in train_metrics.items():
                history['stage1'][f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history['stage1'][f'val_{k}'].append(v)
            
            # Early stopping check
            if self.stage1_trainer.early_stopping(val_metrics['loss']):
                logger.info("Early stopping cho Stage 1")
                break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.stage1_trainer.save_checkpoint(epoch, val_metrics)
        
        # Train Stage 2: Classification model
        logger.info("\n=== Bắt đầu huấn luyện Stage 2: Classification Model ===")
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            train_metrics = self.stage2_trainer.train_epoch()
            val_metrics = self.stage2_trainer.validate()
            
            # Cập nhật history
            for k, v in train_metrics.items():
                history['stage2'][f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history['stage2'][f'val_{k}'].append(v)
            
            # Early stopping check
            if self.stage2_trainer.early_stopping(val_metrics['loss']):
                logger.info("Early stopping cho Stage 2")
                break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.stage2_trainer.save_checkpoint(epoch, val_metrics)
        
        return history

def train_two_stage_model(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: Optional[TrainingConfig] = None
) -> Tuple[TwoStageClassifier, Dict[str, Dict[str, List[float]]]]:
    """Hàm helper để khởi tạo và train toàn bộ hệ thống"""
    if train_config is None:
        train_config = TrainingConfig()
    
    # Khởi tạo data module
    data_module = AnimalDataModule(data_config)
    
    # Tạo model
    device = torch.device(train_config.device)
    model = TwoStageClassifier(model_config).to(device)
    
    # Khởi tạo trainer và train
    trainer = TwoStageTrainer(model, data_module, train_config)
    history = trainer.train()
    
    return model, history

if __name__ == "__main__":
    # Test training pipeline
    from src.config import load_config
    data_config, model_config, train_config, _ = load_config()
    
    model, history = train_two_stage_model(
        data_config,
        model_config,
        train_config
    )
    
    logger.info("Training completed successfully!")