# Hệ thống Phát hiện và Phân loại Động vật

## Cài đặt

1. Tạo môi trường ảo Python:
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

## Chức năng

- Phát hiện đối tượng trong ảnh sử dụng YOLO
- Phân loại động vật thành ăn thịt (carnivore) hoặc ăn cỏ (herbivore)
- Trả về bounding box và confidence score cho mỗi đối tượng
- Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các bbox chồng chéo

## Cách sử dụng

```bash
# Dự đoán trên một ảnh
python main.py predict path/to/image.jpg

# Hiển thị kết quả
python main.py predict path/to/image.jpg --show
```

## Kết quả trả về

```python
{
    'status': 'success',
    'predictions': [
        {
            'bbox': (x1, y1, x2, y2),
            'confidence': 0.95,
            'animal_type': 'carnivore',
            'class_confidence': 0.89
        },
        ...
    ]
}
```

## API Usage

### Running the API Server

1. Install additional dependencies:

```bash
pip install fastapi uvicorn python-multipart aiofiles
```
