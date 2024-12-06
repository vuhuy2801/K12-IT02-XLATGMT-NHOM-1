import logging
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, Optional
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
        self.device = torch.device(config.device)
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
                animal_type = 'carnivore' if pred_class == 0 else 'herbivore'
                
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
            logger.warning(f"Prediction error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'predictions': []
            }
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)}")
            return {
                'status': 'error',
                'message': 'Lỗi không xác định trong quá trình xử lý',
                'predictions': []
            }

def create_predictor(
    model_config: ModelConfig,
    prediction_config: PredictionConfig
) -> Predictor:
    """Tạo và cấu hình predictor"""
    
    try:
        # Khởi tạo model
        model = TwoStageClassifier(model_config)
        model = model.to(prediction_config.device)
        model.eval()
        
        # Tạo predictor
        predictor = Predictor(model, prediction_config)
        logger.info("Đã khởi tạo predictor thành công")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo predictor: {str(e)}")
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