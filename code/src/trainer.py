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
from sklearn.metrics import classification_report

from .config import ModelConfig, DataConfig, TrainingConfig
from .model import AnimalClassifier, TwoStageClassifier
from .data import AnimalDataModule
from .reporting.report_generator import PerformanceReport

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

class Trainer:
    """Training pipeline cho animal classification model"""
    
    def __init__(
        self,
        model: TwoStageClassifier,
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
            self.model.classification_model.parameters(),  # Chỉ train classifier
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=config.mixed_precision)
        
        # Monitoring
        self.writer = SummaryWriter(
            config.tensorboard_dir / f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Tạo thư mục checkpoints
        self.checkpoint_dir = config.checkpoint_dir / "trainer"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Thêm report generator
        self.report_generator = PerformanceReport(config.report_dir)
        
        # Khởi tạo dictionary để lưu history
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'precision': {'carnivore': [], 'herbivore': []},
            'recall': {'carnivore': [], 'herbivore': []},
            'f1': {'carnivore': [], 'herbivore': []},
            'learning_rate': []
        }
        
        logger.info(f"Khởi tạo trainer cho {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train một epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Thêm tracking cho batch metrics
        batch_losses = []
        batch_accuracies = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track batch metrics
            batch_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_accuracies.append(100. * correct / total)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': np.mean(batch_losses),
                'acc': np.mean(batch_accuracies)
            })
            
            # Log to TensorBoard
            step = batch_idx + len(self.train_loader) * self.current_epoch
            self.writer.add_scalar('train_loss', loss.item(), step)
            self.writer.add_scalar('train_accuracy', 
                                 100. * correct / total, step)
        
        # Calculate epoch metrics
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracies)
        
        # Track learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.metrics['learning_rate'].append(current_lr)
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], List, List]:
        """Validation với detailed metrics"""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images, targets = images.to(self.device), targets.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Collect targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(all_targets, all_predictions, 
                                    target_names=['carnivore', 'herbivore'],
                                    output_dict=True)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': report['accuracy'],
            'carnivore_precision': report['carnivore']['precision'],
            'carnivore_recall': report['carnivore']['recall'],
            'carnivore_f1': report['carnivore']['f1-score'],
            'herbivore_precision': report['herbivore']['precision'],
            'herbivore_recall': report['herbivore']['recall'],
            'herbivore_f1': report['herbivore']['f1-score']
        }
        
        return metrics, all_targets, all_predictions
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'training_history': self.metrics
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint for epoch {epoch}")

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log training và validation metrics"""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Carnivore F1: {val_metrics['carnivore_f1']:.4f} | "
            f"Herbivore F1: {val_metrics['herbivore_f1']:.4f}"
        )

    def train(self, num_epochs: int):
        """Training loop với reporting"""
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Lưu ground truth và predictions cho validation set
        val_true = []
        val_pred = []
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics, epoch_true, epoch_pred = self.validate()
            
            # Cập nhật history
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['train_accuracy'].append(train_metrics['accuracy'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['val_accuracy'].append(val_metrics['accuracy'])
            
            # Cập nhật per-class metrics
            for class_name in ['carnivore', 'herbivore']:
                self.metrics['precision'][class_name].append(val_metrics[f'{class_name}_precision'])
                self.metrics['recall'][class_name].append(val_metrics[f'{class_name}_recall'])
                self.metrics['f1'][class_name].append(val_metrics[f'{class_name}_f1'])
            
            # Lưu ground truth và predictions cho reporting
            val_true.extend(epoch_true)
            val_pred.extend(epoch_pred)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Generate plots theo interval
            if self.config.generate_plots and epoch % self.config.plot_metrics_interval == 0:
                self.report_generator.generate_training_report(
                    history=self.metrics,
                    y_true=val_true,
                    y_pred=val_pred,
                    classes=['carnivore', 'herbivore'],
                    experiment_name=f"epoch_{epoch}"
                )
            
            # Save checkpoint nếu model tốt hơn
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics)
                
            # Early stopping
            if epoch - best_epoch >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Generate final report
        final_report_dir = self.report_generator.generate_training_report(
            history=self.metrics,
            y_true=val_true,
            y_pred=val_pred,
            classes=['carnivore', 'herbivore'],
            experiment_name="final"
        )
        
        logger.info(f"Training completed. Final report generated at: {final_report_dir}")
        return self.metrics

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
    
    # Khởi tạo model với cả 2 stages
    device = torch.device(train_config.device)
    model = TwoStageClassifier(model_config).to(device)
    
    # Khởi tạo trainer và train
    trainer = Trainer(model, data_module.train_loader, data_module.val_loader, train_config)
    history = trainer.train(train_config.num_epochs)
    
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