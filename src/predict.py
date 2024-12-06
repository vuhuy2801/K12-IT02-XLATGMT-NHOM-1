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
    """Class xử lý prediction với mô hình 2 giai đoạn"""
    
    def __init__(self, model_config: ModelConfig, prediction_config: PredictionConfig):
        self.model_config = model_config
        self.prediction_config = prediction_config
        self.device = torch.device(prediction_config.device)
        
        # Khởi tạo model
        try:
            self.model = self._load_model()
            self.model.eval()
            logger.info(f"Đã load model thành công trên {self.device}")
        except Exception as e:
            logger.error(f"Lỗi khi load model: {str(e)}")
            raise PredictionException("Không thể khởi tạo model")
        
        # Định nghĩa transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self) -> TwoStageClassifier:
        """Load model từ checkpoint"""
        model = TwoStageClassifier(self.model_config)
        
        # Load stage 1 weights
        if self.prediction_config.stage1_model_path:
            stage1_checkpoint = torch.load(
                self.prediction_config.stage1_model_path,
                map_location=self.device
            )
            model.validation_model.load_state_dict(stage1_checkpoint['model_state_dict'])
            logger.info(f"Đã load Stage 1 model từ {self.prediction_config.stage1_model_path}")
        
        # Load stage 2 weights
        if self.prediction_config.stage2_model_path:
            stage2_checkpoint = torch.load(
                self.prediction_config.stage2_model_path,
                map_location=self.device
            )
            model.classification_model.load_state_dict(stage2_checkpoint['model_state_dict'])
            logger.info(f"Đã load Stage 2 model từ {self.prediction_config.stage2_model_path}")
        
        return model.to(self.device)
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Tiền xử lý ảnh đầu vào"""
        try:
            image = Image.open(image_path).convert('RGB')
            transformed_image = self.transform(image)
            return transformed_image.unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            raise PredictionException(f"Không thể xử lý ảnh: {str(e)}")

    def _validate_input(self, image_path: Union[str, Path]) -> None:
        """Kiểm tra tính hợp lệ của ảnh đầu vào"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise PredictionException(f"Không tìm thấy file {image_path}")
            
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            raise PredictionException(f"Định dạng file không được hỗ trợ: {image_path.suffix}")
            
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception as e:
            raise PredictionException(f"File không phải ảnh hợp lệ: {str(e)}")

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Dự đoán cho một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh cần dự đoán
            
        Returns:
            Dict chứa kết quả dự đoán với các thông tin:
            - status: Trạng thái đoán ("success", "invalid_input", "low_confidence")
            - message: Thông báo chi tiết
            - confidence: Độ tin cậy của dự đoán
            - prediction: Nhãn dự đoán (nếu thành công)
            - time_taken: Thời gian xử lý (ms)
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_input(image_path)
            
            # Preprocess
            image_tensor = self._preprocess_image(image_path)
            
            # Predict
            result = self.model(image_tensor)
            
            # Add timing info
            result['time_taken'] = round((time.time() - start_time) * 1000, 2)  # ms
            
            return result
            
        except PredictionException as e:
            logger.warning(f"Lỗi dự đoán hợp lệ: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'time_taken': round((time.time() - start_time) * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Lỗi không mong đợi: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Có lỗi xảy ra trong quá trình xử lý',
                'time_taken': round((time.time() - start_time) * 1000, 2)
            }

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Dự đoán cho nhiều ảnh"""
        return [self.predict(image_path) for image_path in image_paths]

def create_predictor(
    model_config: Optional[ModelConfig] = None,
    prediction_config: Optional[PredictionConfig] = None
) -> Predictor:
    """Helper function để tạo predictor"""
    if model_config is None or prediction_config is None:
        from src.config import load_config
        _, model_config, _, prediction_config = load_config()
    
    return Predictor(model_config, prediction_config)

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
        print(f"Dự đoán: {result['prediction']}")
        print(f"Độ tin cậy: {result['confidence']:.2%}")
    print(f"Thông báo: {result['message']}")
    print(f"Thời gian xử lý: {result['time_taken']}ms")