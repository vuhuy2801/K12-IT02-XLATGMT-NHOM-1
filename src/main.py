import argparse
import logging
import time
import cv2
from PIL import Image
from pathlib import Path

from src.config import load_config
from src.predict import create_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        image = Image.open(args.image_path)
        logger.info(f"Đã load ảnh: {args.image_path} ({image.size})")
    except Exception as e:
        logger.error(f"Lỗi khi load ảnh: {str(e)}")
        return
        
    # Thực hiện prediction với timing
    logger.info("Bắt đầu prediction...")
    start_time = time.time()
    result = predictor.predict(image)
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

def main():
    parser = argparse.ArgumentParser(description='Animal Detection & Classification')
    subparsers = parser.add_subparsers(dest='command')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on an image')
    predict_parser.add_argument('image_path', type=str, help='Path to image file')
    predict_parser.add_argument('--show', action='store_true', help='Show results')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 