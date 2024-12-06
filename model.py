from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    num_classes: int = 90
    dropout1: float = 0.5
    dropout2: float = 0.3
    hidden_dim: int = 512
    pretrained: bool = True
    freeze_backbone: bool = True

class CustomHead(nn.Module):
    """Custom classification head for the model"""
    
    def __init__(self, in_features: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.head = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(p=config.dropout1)),
            ('fc1', nn.Linear(in_features, config.hidden_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(p=config.dropout2)),
            ('fc2', nn.Linear(config.hidden_dim, config.num_classes))
        ]))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights using Xavier/Glorot initialization"""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

class AnimalClassifier(nn.Module):
    """ResNet50-based model for animal classification"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=config.pretrained)
        
        if config.freeze_backbone:
            self._freeze_backbone()
        
        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = CustomHead(in_features, config)
        
        logger.info(f"Initialized AnimalClassifier with {config}")
    
    def _freeze_backbone(self):
        """Freeze all backbone layers except final blocks"""
        named_params = list(self.backbone.named_parameters())
        unfreeze_keywords = ['layer4', 'layer3']
        
        for name, param in named_params:
            should_freeze = not any(keyword in name for keyword in unfreeze_keywords)
            param.requires_grad = not should_freeze
            
        frozen_params = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.backbone.parameters())
        logger.info(f"Frozen {frozen_params}/{total_params} backbone parameters")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """Get information about trainable parameters"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append({
                    'name': name,
                    'shape': list(param.shape),
                    'params': param.numel()
                })
        return trainable_params

def create_model(device: torch.device, config: Optional[ModelConfig] = None) -> AnimalClassifier:
    """Factory function to create and initialize the model"""
    if config is None:
        config = ModelConfig()
    
    model = AnimalClassifier(config)
    model = model.to(device)
    
    # Log model summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"out of {total_params:,} total parameters")
    
    return model

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create config with custom parameters
    config = ModelConfig(
        num_classes=90,
        dropout1=0.5,
        dropout2=0.3,
        hidden_dim=512,
        pretrained=True,
        freeze_backbone=True
    )
    
    # Create model
    model = create_model(device, config)
    
    # Print trainable parameters info
    trainable_params = model.get_trainable_params()
    for param_info in trainable_params:
        logger.info(f"Trainable layer: {param_info['name']}, "
                   f"Shape: {param_info['shape']}, "
                   f"Parameters: {param_info['params']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}") 