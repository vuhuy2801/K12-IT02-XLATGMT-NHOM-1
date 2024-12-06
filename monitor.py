from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import json
import psutil
import GPUtil

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
    classification_report
)
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.model import AnimalClassifier
from src.data import AnimalDataModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    output_dir: Path
    tensorboard_dir: Path
    log_file: Path
    metrics_file: Path
    system_metrics_interval: int = 60  # seconds
    batch_metrics_logging: bool = True
    save_confusion_matrix: bool = True
    track_gpu_metrics: bool = torch.cuda.is_available()

class PerformanceMetrics:
    """Handles calculation and tracking of model performance metrics"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.batch_times = []
        self.inference_times = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, 
               probs: torch.Tensor, batch_time: float, 
               inference_time: float):
        """Update metrics with batch results"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
        self.batch_times.append(batch_time)
        self.inference_times.append(inference_time)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Calculate basic metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Calculate per-class metrics
        class_report = classification_report(
            targets, predictions, output_dict=True
        )
        
        # Calculate timing metrics
        avg_batch_time = np.mean(self.batch_times)
        avg_inference_time = np.mean(self.inference_times)
        
        return {
            'accuracy': np.mean(predictions == targets),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'class_report': class_report,
            'avg_batch_time': avg_batch_time,
            'avg_inference_time': avg_inference_time,
            'total_samples': len(predictions)
        }

class SystemMetrics:
    """Handles system resource monitoring"""
    
    def __init__(self, track_gpu: bool = True):
        self.track_gpu = track_gpu
        self.metrics_history = []
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
        }
        
        if self.track_gpu:
            try:
                gpu = GPUtil.getGPUs()[0]  # Get first GPU
                metrics.update({
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_percent': gpu.memoryUtil * 100
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {str(e)}")
        
        return metrics
    
    def update(self):
        """Update metrics history"""
        current_metrics = self.get_current_metrics()
        current_metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(current_metrics)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of system metrics"""
        df = pd.DataFrame(self.metrics_history)
        summary = {}
        
        for column in df.columns:
            if column != 'timestamp':
                summary[column] = {
                    'mean': df[column].mean(),
                    'max': df[column].max(),
                    'min': df[column].min(),
                    'std': df[column].std()
                }
        
        return summary

class ModelMonitor:
    """Main monitoring and evaluation system"""
    
    def __init__(
        self,
        model: AnimalClassifier,
        data_module: AnimalDataModule,
        config: MonitoringConfig
    ):
        self.model = model
        self.data_module = data_module
        self.config = config
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.writer = SummaryWriter(self.config.tensorboard_dir)
        self.performance_metrics = PerformanceMetrics(
            num_classes=model.config.num_classes
        )
        self.system_metrics = SystemMetrics(
            track_gpu=config.track_gpu_metrics
        )
        
        # Set up file logging
        self.file_handler = logging.FileHandler(self.config.log_file)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.file_handler)
    
    @torch.no_grad()
    def evaluate_model(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model performance"""
        self.model.eval()
        self.performance_metrics.reset()
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            batch_start = time.time()
            
            # Move to device
            images = images.to(next(self.model.parameters()).device)
            targets = targets.to(next(self.model.parameters()).device)
            
            # Inference
            inference_start = time.time()
            outputs = self.model(images)
            inference_time = time.time() - inference_start
            
            # Get predictions and probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            batch_time = time.time() - batch_start
            self.performance_metrics.update(
                predictions, targets, probabilities,
                batch_time, inference_time
            )
            
            # Log batch metrics
            if self.config.batch_metrics_logging:
                self.writer.add_scalar(
                    'batch/inference_time',
                    inference_time,
                    batch_idx
                )
            
            # Update system metrics
            if batch_idx % 10 == 0:  # Every 10 batches
                self.system_metrics.update()
        
        # Compute final metrics
        metrics = self.performance_metrics.compute_metrics()
        
        # Log metrics to TensorBoard
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'metrics/{name}', value)
        
        # Save confusion matrix plot
        if self.config.save_confusion_matrix:
            self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = self.config.output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        plt.close()
    
    def generate_report(self, metrics: Dict[str, Any]):
        """Generate and save evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_metrics': metrics,
            'system_metrics': self.system_metrics.get_metrics_summary()
        }
        
        # Save report as JSON
        report_path = self.config.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Log summary
        logger.info("=== Evaluation Report ===")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s")
        logger.info(f"Report saved to: {report_path}")
    
    def close(self):
        """Clean up resources"""
        self.writer.close()
        self.file_handler.close()
        logger.removeHandler(self.file_handler)

if __name__ == "__main__":
    from model import ModelConfig
    from src.data import DataConfig
    
    # Configurations
    data_config = DataConfig(
        data_dir=Path("dataset")
    )
    
    model_config = ModelConfig(
        num_classes=90
    )
    
    monitoring_config = MonitoringConfig(
        output_dir=Path("monitoring/output"),
        tensorboard_dir=Path("monitoring/tensorboard"),
        log_file=Path("monitoring/logs/eval.log"),
        metrics_file=Path("monitoring/metrics.json")
    )
    
    # Set up components
    model = AnimalClassifier(model_config)
    data_module = AnimalDataModule(data_config)
    
    # Initialize monitor
    monitor = ModelMonitor(model, data_module, monitoring_config)
    
    # Run evaluation
    _, _, test_loader = data_module.get_data_loaders()
    metrics = monitor.evaluate_model(test_loader)
    
    # Generate report
    monitor.generate_report(metrics)
    
    # Clean up
    monitor.close() 