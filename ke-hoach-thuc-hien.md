# kế hoạch thực hiện

# Giai đoạn nghiên cứu và chuẩn bị

- Dataset được lựa chọn: Animals-90
    - 90 classes động vật khác nhau
    - Khoảng 5,400 ảnh
    - Nguồn: Kaggle ([https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals))
    - Lý do chọn:
        - Dataset đa dạng với nhiều loài động vật
        - Số lượng ảnh đủ lớn cho việc training
        - Chất lượng ảnh tốt và đã được gán nhãn
        - Phù hợp với bài toán phân loại động vật
- Phân tích và lựa chọn mô hình CNN:
    - So sánh các mô hình tiềm năng:
        1. VGG16:
            - Ưu điểm: Kiến trúc đơn giản, dễ hiểu
            - Nhược điểm: Số lượng tham số lớn, training chậm
        2. ResNet50:
            - Ưu điểm: Giải quyết vấn đề vanishing gradient
            - Nhược điểm: Phức tạp hơn trong việc triển khai
        3. MobileNet:
            - Ưu điểm: Nhẹ, nhanh, phù hợp deploy
            - Nhược điểm: Độ chính xác có thể thấp hơn
    - Quyết định: Chọn ResNet50 vì:
        - Cân bằng giữa độ chính xác và tốc độ
        - Có sẵn pretrained weights trên ImageNet
        - Phù hợp với dataset size của chúng ta
        - Có khả năng transfer learning tốt

# Kế hoạch xử lý dữ liệu

- Tiền xử lý ảnh:
    - Resize về kích thước 224x224 (chuẩn của ResNet)
    - Chuẩn hóa pixel values về range [-1, 1]
    - Kiểm tra và xử lý ảnh bị lỗi/nhiễu
- Data Augmentation:
    - Horizontal flip (p=0.5)
    - Random rotation (±15 degrees)
    - Random brightness & contrast adjustment
    - Lý do cho mỗi augmentation:
        - Flip: Tăng khả năng nhận dạng động vật ở các góc nhìn khác nhau
        - Rotation: Mô phỏng góc chụp thực tế
        - Brightness/Contrast: Tăng khả năng thích nghi với điều kiện ánh sáng
- Phân chia dataset:
    - Training: 75% (~4,050 ảnh)
    - Validation: 15% (~810 ảnh)
    - Test: 10% (~540 ảnh)

# Kế hoạch huấn luyện model

- Transfer Learning Strategy:
    - Sử dụng ResNet50 pretrained trên ImageNet
    - Freeze các base layers
    - Thay đổi fully connected layer cuối cùng (90 classes)
    - Fine-tuning các layers cuối trong quá trình training
- Training Parameters:
    - Batch size: 32 (phù hợp với GPU memory)
    - Learning rate: 0.001 với scheduler
    - Optimizer: Adam
    - Loss function: Cross Entropy Loss
    - Epochs: 50 (với early stopping)
- Monitoring và Optimization:
    - Sử dụng TensorBoard để theo dõi:
        - Training/validation loss
        - Accuracy metrics
        - Learning rate changes
    - Early stopping với patience=5
    - Learning rate reduction khi plateau

# Kế hoạch đánh giá model

- Metrics đánh giá:
    - Accuracy: Metric chính để đánh giá tổng thể
    - Precision: Đánh giá độ chính xác của từng class
    - Recall: Đánh giá khả năng phát hiện đúng của từng class
    - F1-score: Cân bằng giữa precision và recall
    - Confusion Matrix: Phân tích chi tiết performance trên từng class
- Phân tích lỗi:
    - Xác định các classes có performance kém
    - Phân tích nguyên nhân:
        - Thiếu dữ liệu training
        - Similarity giữa các classes
        - Chất lượng ảnh không đồng đều
    - Đề xuất cải thiện:
        - Thu thập thêm dữ liệu cho classes yếu
        - Điều chỉnh augmentation strategy
        - Fine-tuning model parameters

# Kế hoạch xây dựng ứng dụng demo

- Backend (Flask API):
    - Endpoints:
        - /upload: Nhận ảnh từ user
        - /predict: Xử lý và trả về kết quả
    - Xử lý ảnh:
        - Validate input
        - Resize và normalize
        - Inference với trained model
        - Return predictions với confidence scores
- Frontend (HTML/CSS/JavaScript):
    - Giao diện:
        - Upload zone cho ảnh
        - Preview ảnh đã chọn
        - Loading indicator
        - Hiển thị kết quả:
            - Top-3 predictions với confidence
            - Visualize attention maps (optional)
    - Responsive design cho mobile/desktop
- Deployment:
    - Docker containerization
    - Requirements:
        - Python 3.8+
        - PyTorch
        - Flask
        - OpenCV
    - Environment variables cho configuration
    - Logging system cho monitoring

# Kế hoạch viết báo cáo và tài liệu

- Cấu trúc báo cáo:
    - Abstract
    - Introduction
    - Methodology
        - Dataset analysis
        - Model architecture
        - Training process
    - Results and Discussion
        - Performance metrics
        - Error analysis
        - Limitations
    - Conclusion and Future Work
- Tài liệu kỹ thuật:
    - Setup guide
    - API documentation
    - Model architecture details
    - Training/inference pipelines
    - Troubleshooting guide
- Slides thuyết trình:
    - Overview của project
    - Key technical decisions
    - Demo của application
    - Results và insights
    - Q&A preparation