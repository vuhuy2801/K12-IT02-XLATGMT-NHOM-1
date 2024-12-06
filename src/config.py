from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

@dataclass
class DataConfig:
    """Cấu hình cho data pipeline"""
    data_dir: Path
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
class ModelConfig:
    """Cấu hình cho model"""
    # Stage 1: Animal Validation
    validation_threshold: float = 0.8
    stage1_model: str = "resnet18"
    
    # Stage 2: Carnivore/Herbivore Classification
    classification_threshold: float = 0.85
    stage2_model: str = "resnet50"
    num_classes: int = 2
    hidden_dim: int = 512
    dropout1: float = 0.5
    dropout2: float = 0.3
    
    # Chung cho cả 2 stage
    pretrained: bool = True
    freeze_backbone: bool = True

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

def load_config():
    """Load và trả về các cấu hình mặc định"""
    data_config = DataConfig(
        data_dir=Path("dataset")
    )
    
    model_config = ModelConfig()
    training_config = TrainingConfig() 
    prediction_config = PredictionConfig(
        stage1_model_path=Path("models/stage1_validation/best.pt"),
        stage2_model_path=Path("models/stage2_classifier/best.pt")
    )
    
    return data_config, model_config, training_config, prediction_config