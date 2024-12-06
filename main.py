import argparse
import logging
from pathlib import Path
import time
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.config import load_config
from src.data import AnimalDataModule
from src.model import TwoStageClassifier
from src.trainer import train_two_stage_model
from src.predict import create_predictor
from src.utils.logging import setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_predictions(image: np.ndarray, predictions: Dict[str, Any]) -> np.ndarray:
    """Vẽ kết quả prediction lên ảnh với nhiều cải tiến"""
    image = image.copy()
    
    if predictions['status'] != 'success':
        # Vẽ thông báo lỗi với style tốt hơn
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.putText(
            image,
            predictions['message'],
            (10, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255),
            2
        )
        return image
        
    # Vẽ thông tin tổng quan
    num_detections = len(predictions['predictions'])
    summary = f"Tìm thấy {num_detections} đối tượng"
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    cv2.putText(
        image,
        summary,
        (10, 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1
    )
        
    # Vẽ mỗi detection với style cải tiến
    for i, pred in enumerate(predictions['predictions'], 1):
        if pred['bbox'] is None:
            continue
            
        # Lấy tọa độ bbox
        x1, y1, x2, y2 = map(int, pred['bbox'])
        # Vẽ bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ confidence score
        conf_text = f"Conf: {pred['confidence']:.2f}"
        cv2.putText(
            image,
            conf_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    
    return image

def predict(args):
    """Chạy prediction trên ảnh"""
    _, model_config, _, prediction_config = load_config()
    
    # Load model
    device = torch.device(prediction_config.device)
    model = TwoStageClassifier(model_config).to(device)
    
    # Load checkpoints
    if prediction_config.stage1_model_path:
        stage1_ckpt = torch.load(prediction_config.stage1_model_path)
        model.validation_model.load_state_dict(stage1_ckpt['model_state_dict'])
    
    if prediction_config.stage2_model_path:
        stage2_ckpt = torch.load(prediction_config.stage2_model_path)
        model.classification_model.load_state_dict(stage2_ckpt['model_state_dict'])
    
    model.eval()
    
    # Load và preprocess ảnh
    try:
        image = Image.open(args.image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
    except Exception as e:
        logger.error(f"Lỗi khi load ảnh: {str(e)}")
        return
    
    # Thực hiện prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        validation_score = outputs['validation'].item()
        
        result = {
            'status': 'success',
            'predictions': []
        }
        
        # Kiểm tra validation score
        if validation_score > model_config.validation_threshold:
            class_probs = torch.softmax(outputs['classification'], dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            confidence = class_probs[0][predicted_class].item()
            
            animal_type = 'carnivore' if predicted_class == 0 else 'herbivore'
            
            result['predictions'].append({
                'animal_type': animal_type,
                'validation_confidence': validation_score,
                'class_confidence': confidence
            })
        else:
            result['status'] = 'no_animal_detected'
            result['message'] = 'Không phát hiện động vật trong ảnh'
    
    # In kết quả
    print("\n=== Kết quả Phân tích ===")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        for pred in result['predictions']:
            print(f"\nLoại động vật: {pred['animal_type'].upper()}")
            print(f"Độ tin cậy validation: {pred['validation_confidence']:.2%}")
            print(f"Độ tin cậy phân loại: {pred['class_confidence']:.2%}")
    else:
        print(f"\nThông báo: {result['message']}")
    
    # Hiển thị ảnh nếu được yêu cầu
    if args.show:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        
        if result['status'] == 'success':
            pred = result['predictions'][0]
            title = f"{pred['animal_type'].upper()}\n"
            title += f"Val Conf: {pred['validation_confidence']:.2%}\n"
            title += f"Class Conf: {pred['class_confidence']:.2%}"
            plt.title(title)
            
        plt.show()
    
    return result

def train(args):
    """Huấn luyện model"""
    logger.info("Bắt đầu quá trình training...")
    
    # Load configs
    data_config, model_config, train_config, _ = load_config()
    
    # Cập nhật config từ command line args
    if args.batch_size:
        data_config.batch_size = args.batch_size
        logger.info(f"Sử dụng batch size: {args.batch_size}")
    if args.epochs:
        train_config.num_epochs = args.epochs
        logger.info(f"Sử dụng epochs: {args.epochs}")
        
    try:
        # Khởi tạo data module
        data_module = AnimalDataModule(data_config)
        
        # Train model
        model, history = train_two_stage_model(
            data_config,
            model_config,
            train_config
        )
        
        logger.info("Huấn luyện hoàn tất!")
        
        # Log training history
        logger.info("Training metrics:")
        for metric, values in history.items():
            logger.info(f"{metric}: {values[-1]:.4f}")
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Animal Classification CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction')
    predict_parser.add_argument('image_path', type=str, help='Path to image')
    predict_parser.add_argument('--show', action='store_true', help='Show result image')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    # Setup logging trước khi chạy
    setup_logging()
    main()