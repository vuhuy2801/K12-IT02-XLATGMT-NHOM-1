import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data import AnimalDataModule
from src.model import TwoStageClassifier
from src.trainer import train_two_stage_model
from src.predict import create_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    """Huấn luyện model"""
    data_config, model_config, train_config, _ = load_config()
    
    # Cập nhật config từ command line args
    if args.batch_size:
        data_config.batch_size = args.batch_size
    if args.epochs:
        train_config.num_epochs = args.epochs
        
    model, history = train_two_stage_model(
        data_config,
        model_config,
        train_config
    )
    logger.info("Huấn luyện hoàn tất!")

def predict(args):
    """Chạy prediction trên ảnh test"""
    _, model_config, _, prediction_config = load_config()
    predictor = create_predictor(model_config, prediction_config)
    
    result = predictor.predict(args.image_path)
    print("\nKết quả dự đoán:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Dự đoán: {result['prediction']}")
        print(f"Độ tin cậy: {result['confidence']:.2%}")
    print(f"Thông báo: {result['message']}")
    print(f"Thời gian xử lý: {result['time_taken']}ms")

def main():
    parser = argparse.ArgumentParser(description='Animal Classification System')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Huấn luyện model')
    train_parser.add_argument('--batch-size', type=int, help='Kích thước batch')
    train_parser.add_argument('--epochs', type=int, help='Số epochs')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Chạy dự đoán')
    predict_parser.add_argument('image_path', type=str, help='Đường dẫn đến ảnh')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()