# Hệ Thống Nhận Dạng và Phân Loại Động Vật Ăn Thịt và Ăn Cỏ Trong Ảnh

## Giới Thiệu
Học phần Xử Lý Ảnh và Thị Giác Máy Tính - Trường Đại học Công Nghệ Đông Á

**Nhóm thực hiện:**
- Vũ Quang Huy - 20210611 - DCCNTT12.10.2
- Lò Tiến Anh - 20210526 - DCCNTT12.10.2

**Giảng viên hướng dẫn:** Lương Thị Hồng Lan

## Mô Tả Dự Án
Dự án xây dựng một hệ thống hai giai đoạn để phát hiện và phân loại động vật trong ảnh thành hai nhóm: động vật ăn thịt và động vật ăn cỏ. Hệ thống sử dụng kết hợp YOLO cho việc phát hiện đối tượng và mạng CNN tùy chỉnh cho việc phân loại.

## Tính Năng
- Phát hiện đối tượng động vật trong ảnh
- Phân loại động vật thành hai nhóm: ăn thịt và ăn cỏ
- Cung cấp điểm tin cậy cho mỗi dự đoán
- API REST cho dự đoán thời gian thực
- Giao diện trực quan để hiển thị kết quả

## Cấu Trúc Dự Án

### Cấu Trúc Thư Mục
```
src/
├── config/                 # Cấu hình cho model và training
├── data/                   # Xử lý và load dữ liệu
├── model/                  # Định nghĩa kiến trúc model
├── predict/                # Logic cho prediction
├── trainer/                # Logic cho training
├── utils/                  # Các utility functions
└── reporting/             # Tạo báo cáo và visualization

frontend/
├── public/                # Static files
├── src/
│   ├── app/              # Next.js app router
│   ├── components/       # React components
│   └── styles/          # CSS styles
└── package.json         # Dependencies
```

### Dataset
```
dataset/
├── train/                  # Dữ liệu huấn luyện (70%)
│   ├── carnivores/        # Động vật ăn thịt
│   └── herbivores/        # Động vật ăn cỏ
├── val/                    # Dữ liệu kiểm định (15%)
│   ├── carnivores/
│   └── herbivores/
└── test/                   # Dữ liệu kiểm tra (15%)
    ├── carnivores/
    └── herbivores/
```

Tổng số ảnh: 5,400
- Train: 3,780 ảnh
- Validation: 810 ảnh  
- Test: 810 ảnh

## Kiến Trúc Hệ Thống

### Module Tiền Xử Lý Input
- Chuẩn hóa kích thước ảnh về 224x224
- Chuẩn hóa giá trị pixel về khoảng [0,1]
- Data augmentation khi training

### Module Phát Hiện Đối Tượng (YOLO)
- Sử dụng mạng neural hiện đại để nhận diện vị trí của đối tượng trong ảnh
- Có khả năng phát hiện nhiều đối tượng cùng lúc với độ chính xác cao
- Hoạt động tốt với các kích thước đối tượng khác nhau trong ảnh

### Module Phân Loại (CNN) 
- Phân tích chi tiết vùng ảnh chứa động vật đã được phát hiện
- Học các đặc điểm đặc trưng để phân biệt động vật ăn thịt và ăn cỏ
- Đưa ra kết quả phân loại kèm theo độ tin cậy của dự đoán

## Cài Đặt và Sử Dụng

### Yêu Cầu Hệ Thống
- Python 3.8+
- PyTorch & TorchVision
- YOLO (Ultralytics)
- FastAPI
- OpenCV
- Các thư viện khác trong requirements.txt

### Cài Đặt
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy Ứng Dụng
```bash
# Chạy API server
python run_api.py

# Chạy Frontend

# Di chuyển vào thư mục frontend
cd frontend

# Chạy ở môi trường development
npm run dev

# Dự đoán một ảnh
python main.py predict path/to/image.jpg

# Hiển thị kết quả trực quan
python main.py predict path/to/image.jpg --show
```

## Kết Quả Thực Nghiệm
- Độ chính xác huấn luyện: 98.30%
- Độ chính xác kiểm định: 81.00%
- Thời gian huấn luyện: ~2 giờ
- Dataset: 5,400 hình ảnh từ 90 loài động vật khác nhau (Kaggle)

## Hướng Phát Triển
- Mở rộng số lượng lớp phân loại
- Cải thiện độ chính xác cho đối tượng nhỏ
- Tối ưu hóa tốc độ xử lý
- Phát triển giao diện người dùng web


## Lời Cảm Ơn
Nhóm xin gửi lời cảm ơn chân thành đến cô Lương Thị Hồng Lan đã tận tình hướng dẫn và hỗ trợ trong quá trình thực hiện đề tài.
