import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Chia dataset thành các tập train, validation và test
    
    Parameters:
    - source_dir: Thư mục chứa toàn bộ dữ liệu
    - train_dir: Thư mục đích cho tập train
    - val_dir: Thư mục đích cho tập validation
    - test_dir: Thư mục đích cho tập test
    - train_ratio: Tỷ lệ cho tập train (mặc định: 0.7)
    - val_ratio: Tỷ lệ cho tập validation (mặc định: 0.15)
    - test_ratio: Tỷ lệ cho tập test (mặc định: 0.15)
    """
    
    # Tạo các thư mục đích nếu chưa tồn tại
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Lấy danh sách tất cả các file trong thư mục nguồn
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Chia thành train và temp
    train_files, temp_files = train_test_split(all_files, 
                                             train_size=train_ratio,
                                             random_state=42)
    
    # Chia temp thành validation và test
    relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files,
                                           train_size=relative_ratio,
                                           random_state=42)
    
    # Copy files vào các thư mục tương ứng
    for file_name in train_files:
        shutil.copy2(os.path.join(source_dir, file_name),
                    os.path.join(train_dir, file_name))
    
    for file_name in val_files:
        shutil.copy2(os.path.join(source_dir, file_name),
                    os.path.join(val_dir, file_name))
    
    for file_name in test_files:
        shutil.copy2(os.path.join(source_dir, file_name),
                    os.path.join(test_dir, file_name))
    
    print(f"Tổng số files: {len(all_files)}")
    print(f"Số files trong tập train: {len(train_files)}")
    print(f"Số files trong tập validation: {len(val_files)}")
    print(f"Số files trong tập test: {len(test_files)}")

if __name__ == "__main__":
    # Thay đổi các đường dẫn này theo cấu trúc thư mục của bạn
    SOURCE_DIR = "dataset"
    TRAIN_DIR = "dataset/train"
    VAL_DIR = "dataset/val"
    TEST_DIR = "dataset/test"
    
    split_dataset(SOURCE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR) 