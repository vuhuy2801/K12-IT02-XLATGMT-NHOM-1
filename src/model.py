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
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass returns only logits for training
        """
        outputs = self.model(x)
        return outputs
    
    def get_prediction_with_confidence(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Get both prediction and confidence score (for inference)
        """
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
        
        return outputs, confidence

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
    """Mô hình 2 giai đoạn: YOLO Detection + Classification"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Khởi tạo các models
        self.validation_model = AnimalValidationModel(config)
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
        
        # Stack tất cả crops thành một batch và squeeze để bỏ dimension thừa
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