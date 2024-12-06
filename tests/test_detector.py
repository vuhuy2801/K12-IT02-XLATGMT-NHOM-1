import unittest
from PIL import Image
import numpy as np
from src.detector import YOLODetector
from src.config import YOLOConfig

class TestYOLODetector(unittest.TestCase):
    def setUp(self):
        self.config = YOLOConfig()
        self.detector = YOLODetector(self.config)
        
    def test_model_loading(self):
        self.assertIsNotNone(self.detector.model)
        
    def test_detection(self):
        # Create a test image
        image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        
        # Run detection
        detections = self.detector.detect(image)
        
        # Check output format
        for det in detections:
            self.assertEqual(len(det.bbox), 4)
            self.assertTrue(0 <= det.confidence <= 1)
            # Verify bbox coordinates are valid
            x1, y1, x2, y2 = det.bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            
    def test_nms(self):
        # Test NMS với các bbox chồng chéo
        detections = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.9),
            Detection(bbox=(1, 1, 11, 11), confidence=0.8),  # Chồng chéo với bbox đầu tiên
            Detection(bbox=(20, 20, 30, 30), confidence=0.7),  # Không chồng chéo
        ]
        
        filtered = self.detector._apply_nms(detections)
        self.assertEqual(len(filtered), 2)  # Chỉ giữ lại 2 bbox không chồng chéo

if __name__ == '__main__':
    unittest.main() 