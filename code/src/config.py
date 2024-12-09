from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import torch
import numpy as np
import random

def set_random_seed(seed: int = 42) -> None:
    """Thiết lập seed cho tất cả các nguồn ngẫu nhiên để đảm bảo kết quả reproducible
    
    Args:
        seed: Giá trị seed để sử dụng (mặc định: 42)
    """
    # Thiết lập seed cho Python random
    random.seed(seed)
    
    # Thiết lập seed cho NumPy
    np.random.seed(seed)
    
    # Thiết lập seed cho PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu sử dụng multi-GPU
    
    # Một số cài đặt bổ sung cho PyTorch để đảm bảo deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class DataConfig:
    """Cấu hình cho data pipeline"""
    data_dir: Path
    random_seed: int = 42  # Thêm random seed vào config
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.75
    val_split: float = 0.15
    test_split: float = 0.10
    pin_memory: bool = True
    
    # Thêm cấu hình augmentation
    use_augmentation: bool = True
    rotation_degrees: int = 15
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2

@dataclass
class YOLOConfig:
    """Cấu hình cho YOLO detector"""
    model_type: str = "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 10

@dataclass
class ModelConfig:
    """Cấu hình cho model"""
    # Classification parameters
    num_classes: int = 2  # carnivore/herbivore
    hidden_dim: int = 512
    dropout1: float = 0.5
    dropout2: float = 0.3
    
    # Model architecture
    stage2_model: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    
    # YOLO config
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    
    # Detection parameters
    use_detection: bool = True
    min_detection_size: Tuple[int, int] = (32, 32)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    """Cấu hình cho quá trình training"""
    num_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    checkpoint_dir: Path = Path("models")
    tensorboard_dir: Path = Path("runs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Thêm các tham số training
    weight_decay: float = 1e-4
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    
    # Visualization settings
    visualization_interval: int = 5  # Generate plots every N epochs
    save_best_only: bool = True  # Only save checkpoints that improve validation metrics
    monitor_metric: str = 'val_loss'  # Metric to monitor for early stopping/checkpointing
    
    # Additional metrics tracking
    track_gpu_usage: bool = True
    track_memory_usage: bool = True
    track_gradient_norms: bool = True
    
    # Reporting settings
    generate_plots: bool = True
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    plot_metrics_interval: int = 5  # Plot metrics every N epochs
    report_dir: Path = Path("reports")

@dataclass 
class PredictionConfig:
    """Cấu hình cho prediction pipeline"""
    stage1_model_path: Optional[Path] = None
    stage2_model_path: Optional[Path] = None
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Thêm cấu hình cho prediction
    top_k: int = 3
    confidence_threshold: float = 0.5
    timeout: float = 30.0
    show_confidence: bool = True
    confidence_decimal_places: int = 2

def load_config():
    """Load và trả về các cấu hình mặc định"""
    # Thiết lập random seed ngay khi load config
    set_random_seed(42)
    
    data_config = DataConfig(
        data_dir=Path("dataset")
    )
    
    model_config = ModelConfig()
    training_config = TrainingConfig() 
    prediction_config = PredictionConfig(
        stage1_model_path=Path("models/stage1_validation/checkpoint_epoch_19.pt"),
        stage2_model_path=Path("models/stage2_classifier/checkpoint_epoch_9.pt"),
        show_confidence=True,
        confidence_decimal_places=2
    )
    
    return data_config, model_config, training_config, prediction_config