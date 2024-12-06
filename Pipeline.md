# Pipeline

1. Pipeline Xử lý Dữ liệu

```
Ảnh Đầu vào → Tải Dữ liệu → Tiền Xử lý → Tăng cường Dữ liệu → DataLoader

```

Chi tiết các bước:

- Ảnh Đầu vào:
    - Nguồn: Bộ dữ liệu Animals-90
    - Định dạng: JPG/PNG
    - Kích thước: Đa dạng
    - Nhãn: 90 loại động vật
- Tải Dữ liệu:
    - Sử dụng PyTorch ImageFolder
    - Cấu trúc thư mục:
        
        ```
        dataset/
        ├── carnivores/
        |   ├── lion/
        |   ├── fox/
        |   └── ...
        ├── herbivores_omnivores/
        |   ├── antelope/
        |   ├── bat/
            └── ...
        
        ```
        
    - Phân chia dữ liệu: 75/15/10
- Tiền Xử lý:
    - Điều chỉnh kích thước (224x224) - theo chuẩn ResNet50
    - Chuẩn hóa theo thông số ImageNet:
        - trung bình=[0.485, 0.456, 0.406]
        - độ lệch chuẩn=[0.229, 0.224, 0.225]
    - Chuyển đổi sang tensor
- Tăng cường Dữ liệu (chỉ áp dụng cho tập huấn luyện):
    - Lật ngang (xác suất=0.5)
    - Xoay ngẫu nhiên (±15°)
    - Điều chỉnh độ sáng ngẫu nhiên (±20%)
    - Điều chỉnh độ tương phản (±20%)
- DataLoader:
    - Kích thước batch: 32
    - Xáo trộn: Có (cho tập huấn luyện)
    - Số luồng worker: 4
    - Pin memory: Bật (cho GPU)
1. Pipeline Mô hình

```
Tensor Đầu vào → ResNet50 → Đầu Tùy chỉnh → Dự đoán Đầu ra

```

Chi tiết:

- ResNet50 Cơ bản:
    - Đã được huấn luyện trước trên ImageNet
    - Đóng băng tất cả các lớp trừ 2 khối cuối
    - Đặc trưng đầu ra: 2048 chiều
- Đầu Tùy chỉnh:
    
    ```
    2048D → Dropout(0.5) → Linear(2048, 512) → ReLU →
    Dropout(0.3) → Linear(512, 90) → Softmax
    
    ```
    
1. Pipeline Huấn luyện

```
DataLoader → Mô hình → Hàm Mất mát → Lan truyền ngược → Tối ưu hóa → Điều chỉnh LR

```

Chi tiết:

- Vòng lặp Huấn luyện:
    - Lan truyền xuôi:
        
        ```
        ảnh, nhãn = batch
        dự_đoán = mô_hình(ảnh)
        mất_mát = hàm_mất_mát(dự_đoán, nhãn)
        
        ```
        
    - Lan truyền ngược:
        
        ```
        tối_ưu.zero_grad()
        mất_mát.backward()
        tối_ưu.step()
        điều_chỉnh_lr.step()
        
        ```
        
- Vòng lặp Kiểm định:
    
    ```
    mô_hình.eval()
    with torch.no_grad():
        for batch in val_loader:
            dự_đoán = mô_hình(ảnh)
            # Tính các metrics
    
    ```
    
1. Pipeline Dự đoán

```
Ảnh Đầu vào → Tiền xử lý → Dự đoán → Hậu xử lý → Kết quả

```

Chi tiết:

- Xử lý Đầu vào:
    - Kiểm tra định dạng và kích thước
    - Áp dụng tiền xử lý giống huấn luyện
    - Chuyển sang tensor và thêm chiều batch
- Dự đoán:
    
    ```
    mô_hình.eval()
    with torch.no_grad():
        dự_đoán = mô_hình(ảnh)
    
    ```
    
- Hậu xử lý:
    - Top-k dự đoán (k=3)
    - Chuyển logits sang xác suất
    - Ánh xạ chỉ số sang tên lớp
1. Pipeline API

```
Yêu cầu HTTP → Máy chủ Flask → Pipeline Dự đoán → Phản hồi

```

Chi tiết:

- Xử lý Yêu cầu:
    - Kiểm tra file ảnh
    - Trích xuất metadata
    - Đưa vào hàng đợi nếu cần
- Xử lý:
    
    ```
    ảnh = tải_ảnh(request.files['image'])
    dự_đoán = pipeline_dự_đoán(ảnh)
    phản_hồi = định_dạng_phản_hồi(dự_đoán)
    
    ```
    
- Định dạng Phản hồi:
    
    ```json
    {
      "thành_công": true,
      "dự_đoán": [
        {
          "lớp": "mèo",
          "độ_tin_cậy": 0.95
        },
        ...
      ],
      "thời_gian_xử_lý": "0.5s"
    }
    
    ```
    
1. Pipeline Giám sát

```
Mô hình → Thu thập Metrics → TensorBoard → Phân tích

```

Chi tiết:

- Thu thập Metrics:
    - Loss huấn luyện/kiểm định
    - Độ chính xác theo epoch
    - Thay đổi learning rate
    - Sử dụng GPU
    - Thời gian dự đoán
- Ghi Log:
    - Tình trạng hệ thống
    - Yêu cầu API
    - Theo dõi lỗi
    - Metrics hiệu suất

Pipeline này đảm bảo:

- Tính module cao
- Dễ dàng debug và bảo trì
- Khả năng mở rộng
- Giám sát đầy đủ
- Xử lý lỗi chắc chắn

Bạn muốn tôi giải thích chi tiết thêm phần nào không?