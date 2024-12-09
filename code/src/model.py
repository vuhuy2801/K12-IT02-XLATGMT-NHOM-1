from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from PIL import Image
from torchvision import transforms

from .config import ModelConfig
from .detector import YOLODetector, Detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomHead(nn.Module):
    """Custom classification head cho classifier"""
    
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
    """YOLO Detection + Classification Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Khởi tạo classifier
        self.classification_model = AnimalClassifier(config)
        
        # Đảm bảo detector được khởi tạo
        if config.use_detection:
            self.detector = YOLODetector(config.yolo)
        else:
            raise ValueError("Detection must be enabled for this model")
        
        logger.info(f"Initialized TwoStageClassifier with config: {config}")
        
    def detect_and_classify(self, image: Image.Image) -> List[Tuple[Detection, torch.Tensor]]:
        """
        Detect và phân loại động vật trong ảnh
        
        Args:
            image: PIL Image
            
        Returns:
            List[Tuple[Detection, Tensor]]: List các cặp (detection, prediction)
        """
        detections = self.detector.detect(image)
        if not detections:
            return []
        
        # Batch processing thay vì xử lý từng detection
        crops = []
        valid_detections = []
        
        for detection in detections:
            # Crop ảnh theo bbox
            x1, y1, x2, y2 = detection.bbox
            cropped = image.crop((x1, y1, x2, y2))
            
            if (cropped.size[0] >= self.config.min_detection_size[0] and 
                cropped.size[1] >= self.config.min_detection_size[1]):
                # Preprocess image
                tensor = self._preprocess_image(cropped)
                crops.append(tensor)
                valid_detections.append(detection)
        
        if not crops:
            return []
        
        # Stack tất cả crops thành một batch
        batch = torch.cat(crops, dim=0)  # [N, 3, 224, 224]
        
        # Predict một lần cho cả batch
        with torch.no_grad():
            predictions = self.classification_model(batch)  # [N, num_classes]
        
        return list(zip(valid_detections, predictions))
        
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Chuẩn bị ảnh cho classifier"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Chuyển đổi sang tensor và thêm batch dimension
        tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor.to(self.config.device)

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