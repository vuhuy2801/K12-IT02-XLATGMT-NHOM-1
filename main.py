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
from src.reporting import PerformanceReport

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
    
    # Tạo predictor
    try:
        predictor = create_predictor(model_config, prediction_config)
    except Exception as e:
        logger.error(f"Lỗi khi tạo predictor: {str(e)}")
        return
        
    # Load và validate ảnh
    try:
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ {args.image_path}")
        logger.info(f"Đã load ảnh: {args.image_path} ({image.shape})")
    except Exception as e:
        logger.error(f"Lỗi khi load ảnh: {str(e)}")
        return
        
    # Convert BGR sang RGB cho model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Thực hiện prediction với timing
    logger.info("Bắt đầu prediction...")
    start_time = time.time()
    result = predictor.predict(pil_image)
    inference_time = (time.time() - start_time) * 1000
    logger.info(f"Hoàn thành prediction trong {inference_time:.1f}ms")
    
    # In kết quả
    print("\n=== Kết quả Phân tích ===")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"\nTìm thấy {len(result['predictions'])} đối tượng:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n{i}. Đối tượng:")
            print(f"   - Loại: {pred['animal_type'].upper()}")
            print(f"   - Độ tin cậy phát hiện: {pred['confidence']:.1%}")
            print(f"   - Độ tin cậy phân loại: {pred['class_confidence']:.1%}")
            if pred['bbox']:
                x1, y1, x2, y2 = map(int, pred['bbox'])
                print(f"   - Vị trí: ({x1}, {y1}) -> ({x2}, {y2})")
    else:
        print(f"\nThông báo: {result['message']}")
    
    # Vẽ kết quả lên ảnh
    output_image = draw_predictions(image, result)
    
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Lưu ảnh kết quả với timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"pred_{timestamp}_{Path(args.image_path).name}"
    cv2.imwrite(str(output_path), output_image)
    logger.info(f"Đã lưu kết quả tại: {output_path}")
    
    # Hiển thị ảnh nếu được yêu cầu
    if args.show:
        cv2.imshow('Animal Detection & Classification', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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