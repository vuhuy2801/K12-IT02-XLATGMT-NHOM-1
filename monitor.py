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
from PIL import Image
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

from src.model import TwoStageClassifier
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
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.detections = []
        self.batch_times = []
        self.inference_times = []
        self.detection_confidences = []
        self.classification_confidences = []
    
    def update(self, detections: List[Dict], batch_time: float, inference_time: float):
        """Update metrics with batch results"""
        self.detections.extend(detections)
        self.batch_times.append(batch_time)
        self.inference_times.append(inference_time)
        
        for det in detections:
            self.detection_confidences.append(det['confidence'])
            if 'class_confidence' in det:
                self.classification_confidences.append(det['class_confidence'])
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics"""
        metrics = {
            'avg_detection_confidence': np.mean(self.detection_confidences),
            'avg_batch_time': np.mean(self.batch_times),
            'avg_inference_time': np.mean(self.inference_times),
            'total_detections': len(self.detections),
            'detections_per_image': len(self.detections) / len(self.batch_times)
        }
        
        if self.classification_confidences:
            metrics['avg_classification_confidence'] = np.mean(self.classification_confidences)
            
            # Count animal types
            animal_types = [d['animal_type'] for d in self.detections if 'animal_type' in d]
            if animal_types:
                type_counts = pd.Series(animal_types).value_counts()
                metrics['animal_type_distribution'] = type_counts.to_dict()
        
        return metrics

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
        model: TwoStageClassifier,
        data_module: AnimalDataModule,
        config: MonitoringConfig
    ):
        self.model = model
        self.data_module = data_module
        self.config = config
        
        # Create all required directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        # Create logs directory
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.writer = SummaryWriter(self.config.tensorboard_dir)
        self.performance_metrics = PerformanceMetrics()
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
        
        device = next(self.model.parameters()).device
        
        for batch_idx, (images, _) in enumerate(data_loader):
            batch_start = time.time()
            
            # Move to device
            images = images.to(device)
            
            # Convert to PIL images for detection
            pil_images = []
            for img in images:
                # Denormalize if needed
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
                pil_images.append(Image.fromarray(img))
            
            # Process each image
            batch_detections = []
            inference_times = []
            
            for pil_img in pil_images:
                inference_start = time.time()
                results = self.model.detect_and_classify(pil_img)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Convert results to dict format
                detections = []
                for det, logits in results:
                    probs = torch.softmax(logits, dim=0)
                    pred_class = torch.argmax(probs).item()
                    class_confidence = probs[pred_class].item()
                    animal_type = 'Động vật ăn thịt' if pred_class == 0 else 'Động vật ăn cỏ'
                    
                    detection_dict = {
                        'bbox': det.bbox,
                        'confidence': det.confidence,
                        'animal_type': animal_type,
                        'class_confidence': class_confidence
                    }
                    detections.append(detection_dict)
                batch_detections.extend(detections)
            
            # Update metrics
            batch_time = time.time() - batch_start
            self.performance_metrics.update(
                batch_detections,
                batch_time,
                np.mean(inference_times)
            )
            
            # Log batch metrics
            if self.config.batch_metrics_logging:
                self.writer.add_scalar(
                    'batch/inference_time',
                    np.mean(inference_times),
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
        
        return metrics
    
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
        logger.info(f"Average Detection Confidence: {metrics['avg_detection_confidence']:.4f}")
        if 'avg_classification_confidence' in metrics:
            logger.info(f"Average Classification Confidence: {metrics['avg_classification_confidence']:.4f}")
        logger.info(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s")
        logger.info(f"Total Detections: {metrics['total_detections']}")
        logger.info(f"Report saved to: {report_path}")
    
    def close(self):
        """Clean up resources"""
        self.writer.close()
        self.file_handler.close()
        logger.removeHandler(self.file_handler)

if __name__ == "__main__":
    from src.config import load_config
    
    # Create base monitoring directory
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    # Load configs
    data_config, model_config, _, _ = load_config()
    
    monitoring_config = MonitoringConfig(
        output_dir=monitoring_dir / "output",
        tensorboard_dir=monitoring_dir / "tensorboard",
        log_file=monitoring_dir / "logs" / "eval.log",
        metrics_file=monitoring_dir / "metrics.json"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up components
    model = TwoStageClassifier(model_config)
    model = model.to(device)  # Move model to GPU
    data_module = AnimalDataModule(data_config)
    
    # Initialize monitor
    monitor = ModelMonitor(model, data_module, monitoring_config)
    
    try:
        # Run evaluation
        _, _, test_loader = data_module.get_data_loaders()
        metrics = monitor.evaluate_model(test_loader)
        
        # Generate report
        monitor.generate_report(metrics)
        
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")
        raise
    finally:
        # Clean up
        monitor.close() 