# src/__init__.py
from .config import load_config, DataConfig, ModelConfig, TrainingConfig, PredictionConfig
from .data import AnimalDataModule
from .model import TwoStageClassifier
from .trainer import TwoStageTrainer
from .predict import Predictor

__version__ = "0.1.0"