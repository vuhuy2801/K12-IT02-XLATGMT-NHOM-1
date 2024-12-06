import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model import AnimalClassifier, ModelConfig  # Import từ file model.py của bạn

class Predictor:
    def __init__(self, checkpoint_path, model_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnimalClassifier(model_config)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path):
        # Load và xử lý ảnh
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        # Dự đoán
        outputs = self.model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence

# Chạy thử nghiệm
if __name__ == "__main__":
    # Khởi tạo model config (phải giống với lúc training)
    model_config = ModelConfig(
        num_classes=90,  # Số lượng classes của bạn
        pretrained=False
    )

    # Đường dẫn đến checkpoint
    checkpoint_path = "checkpoints/checkpoint_epoch_9.pt"  # Thay đổi theo checkpoint của bạn

    # Khởi tạo predictor
    predictor = Predictor(checkpoint_path, model_config)

    # Thử predict một ảnh
    test_image = "test_images/ff48cfa500ffc1a198ee.jpg"  # Đường dẫn đến ảnh test
    predicted_class, confidence = predictor.predict(test_image)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")