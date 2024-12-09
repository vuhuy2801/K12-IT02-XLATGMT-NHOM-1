from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

from .config import YOLOConfig

@dataclass
class Detection:
    """Class chứa thông tin về một detection"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float

class YOLODetector:
    """YOLO detector cho động vật"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """Load pretrained YOLO model"""
        try:
            model = YOLO(self.config.model_type)
            return model
        except Exception as e:
            raise RuntimeError(f"Không thể load YOLO model: {str(e)}")
            
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Detect objects trong ảnh
        """
        # Convert PIL Image to format YOLO expects
        img_array = np.array(image)
        
        # Run detection với kiến trúc:
        # 1. Backbone: CSPDarknet 
        # 2. Neck: PANet
        # 3. Head: Decoupled Head
        results = self.model(
            img_array,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections
        )
        
        # Process results - chỉ lấy bbox và confidence
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections

    def crop_detection(self, image: Image.Image, detection: Detection) -> Image.Image:
        """
        Crop ảnh theo bounding box của detection
        
        Args:
            image: PIL Image gốc
            detection: Detection object chứa bbox
            
        Returns:
            PIL Image đã được crop
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        return image.crop((x1, y1, x2, y2))

    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        """Áp dụng NMS để loại bỏ các detection chồng chéo"""
        if not detections:
            return []
        
        # Convert to numpy arrays
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Compute areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]