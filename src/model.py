from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from .config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalValidationModel(nn.Module):
    """Stage 1: Kiểm tra xem ảnh có phải động vật không"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load ResNet18 cho stage 1 (nhẹ hơn ResNet50)
        if config.stage1_model == "resnet18":
            base_model = models.resnet18(pretrained=config.pretrained)
        else:
            raise ValueError(f"Model {config.stage1_model} không được hỗ trợ cho validation")
            
        # Freeze backbone nếu cần
        if config.freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False
                
        # Thay đổi lớp cuối để phù hợp với ImageNet animal classes
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.get_animal_classes()))
        )
        
        self.model = base_model
        
    def get_animal_classes(self) -> List[int]:
        """Trả về list các class động vật trong ImageNet"""
        # ImageNet indices cho các class động vật
        return (
            list(range(151, 268)) +  # Mammals
            list(range(270, 374)) +  # Birds
            list(range(38, 68))      # Reptiles, amphibians
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Forward pass trả về logits và confidence score
        """
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
        
        return outputs, confidence.item()

class CustomHead(nn.Module):
    """Custom classification head cho stage 2"""
    
    def __init__(self, in_features: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout1),
            nn.Linear(in_features, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout2),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights using Xavier initialization"""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

class AnimalClassifier(nn.Module):
    """Stage 2: Phân loại động vật (ăn thịt/ăn cỏ)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load ResNet50 cho stage 2
        if config.stage2_model == "resnet50":
            self.backbone = models.resnet50(pretrained=config.pretrained)
        else:
            raise ValueError(f"Model {config.stage2_model} không được hỗ trợ cho classifier")
        
        if config.freeze_backbone:
            self._freeze_backbone()
        
        # Thay đổi classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = CustomHead(in_features, config)
        
        logger.info(f"Initialized AnimalClassifier with {config}")
    
    def _freeze_backbone(self):
        """Đóng băng các layer trừ các block cuối"""
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

class TwoStageClassifier(nn.Module):
    """Mô hình kết hợp cả 2 stage"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.validation_model = AnimalValidationModel(config)
        self.classification_model = AnimalClassifier(config)
    
    def forward(self, x: Tensor) -> Dict[str, Any]:
        """
        Forward pass qua cả 2 stage
        Trả về dict chứa status, prediction và confidence
        """
        # Stage 1: Kiểm tra có phải động vật không
        _, validation_confidence = self.validation_model(x)
        
        if validation_confidence < self.config.validation_threshold:
            return {
                'status': 'invalid_input',
                'message': 'Ảnh này có vẻ kh��ng phải động vật',
                'confidence': validation_confidence,
                'suggestion': 'Vui lòng thử lại với ảnh động vật khác'
            }
        
        # Stage 2: Phân loại loại động vật
        outputs = self.classification_model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        
        if confidence < self.config.classification_threshold:
            return {
                'status': 'low_confidence',
                'message': 'Không thể phân loại với độ tin cậy cao',
                'confidence': confidence,
                'suggestion': 'Vui lòng thử lại với ảnh rõ nét hơn'
            }
        
        prediction_map = {0: 'Động vật ăn thịt', 1: 'Động vật ăn cỏ'}
        predicted_class = prediction_map[prediction.item()]
        
        return {
            'status': 'success',
            'prediction': predicted_class,
            'confidence': confidence,
            'details': f'Động vật này có vẻ là {predicted_class.lower()}'
        }

def create_model(config: ModelConfig, device: torch.device) -> TwoStageClassifier:
    """Hàm tạo model và khởi tạo"""
    model = TwoStageClassifier(config)
    model = model.to(device)
    
    # Log thông tin model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Đã tạo two-stage model với {trainable_params:,} tham số có thể train "
                f"trong tổng số {total_params:,} tham số")
    
    return model