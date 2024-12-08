import logging
from pathlib import Path
from typing import Dict, Any
import time

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .config import ModelConfig, PredictionConfig
from .model import TwoStageClassifier

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionException(Exception):
    """Custom exception cho các lỗi prediction"""
    pass

class Predictor:
    """Class xử lý prediction với nhiều cải tiến"""
    
    def __init__(self, model: TwoStageClassifier, config: PredictionConfig):
        self.model = model
        self.config = config
        self.model.eval()
        
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện prediction trên một ảnh
        
        Args:
            image: PIL Image
            
        Returns:
            Dict chứa kết quả prediction
        """
        try:
            # Validate input
            if not isinstance(image, Image.Image):
                raise PredictionException("Input phải là PIL Image")
                
            # Detect và classify
            results = self.model.detect_and_classify(image)
            
            if not results:
                return {
                    'status': 'success',
                    'message': 'Không tìm thấy đối tượng nào',
                    'predictions': []
                }
            
            # Process results
            predictions = []
            for detection, logits in results:
                # Get classification result
                probs = torch.softmax(logits, dim=0)
                pred_class = torch.argmax(probs).item()
                class_confidence = probs[pred_class].item()
                
                # Determine animal type
                animal_type = 'Động vật ăn thịt' if pred_class == 0 else 'Động vật ăn cỏ'
                
                pred_dict = {
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'animal_type': animal_type,
                    'class_confidence': class_confidence
                }
                predictions.append(pred_dict)
            
            return {
                'status': 'success',
                'predictions': predictions
            }
            
        except PredictionException as e:
            return {
                'status': 'error',
                'message': str(e),
                'predictions': []
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Internal prediction error',
                'predictions': []
            }

def create_predictor(
    model_config: ModelConfig,
    prediction_config: PredictionConfig
) -> Predictor:
    """Tạo predictor từ config"""
    try:
        # Load model
        device = torch.device(prediction_config.device)
        model = TwoStageClassifier(model_config)
        
        # Load stage 2 weights nếu được chỉ định
        if prediction_config.stage2_model_path:
            checkpoint = torch.load(prediction_config.stage2_model_path, map_location=device)
            model.classification_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded stage 2 weights from {prediction_config.stage2_model_path}")
        
        model = model.to(device)
        model.eval()
        
        return Predictor(model, prediction_config)
        
    except Exception as e:
        logger.error(f"Error creating predictor: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Load configs
    from src.config import load_config
    _, model_config, _, prediction_config = load_config()
    
    # Tạo predictor
    predictor = create_predictor(model_config, prediction_config)
    
    # Test với một ảnh
    test_image = "test_images/test.jpg"
    result = predictor.predict(test_image)
    
    print("\nKết quả dự đoán:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Dự đoán: {result['predictions']}")
        print(f"Thông báo: {result['message']}")
        print(f"Thời gian xử lý: {result['time_taken']}ms")