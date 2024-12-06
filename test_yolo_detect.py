from ultralytics import YOLO
import cv2
import numpy as np

def detect_objects_yolo(image_path):
    # Khởi tạo model
    model = YOLO('yolov8n.pt')  # sử dụng model nhỏ nhất cho tốc độ nhanh
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    # Thực hiện dự đoán
    results = model(image)
    
    # Vẽ kết quả
    result_image = image.copy()
    object_count = 0
    
    # Lấy tất cả bounding boxes, bỏ qua class
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            
            # Chỉ vẽ những box có độ tin cậy cao
            if conf > 0.3:  # Điều chỉnh ngưỡng này nếu cần
                object_count += 1
                
                # Vẽ bounding box
                cv2.rectangle(result_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Thêm số thứ tự cho object
                cv2.putText(result_image, 
                          f'Object {object_count}', 
                          (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
    
    return result_image, object_count

# Sử dụng
image_path = 'test_images/cho-nghi-dai.jpg'
result, count = detect_objects_yolo(image_path)

# Hiển thị kết quả
print(f"Đã phát hiện {count} đối tượng")
cv2.imshow('Objects Detected', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu kết quả
cv2.imwrite('result.jpg', result)