# test_libraries.py

import os
import sys
import time
print("Python version:", sys.version)

# Test PyTorch và CUDA
print("\n=== Testing PyTorch and CUDA ===")
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        
        # Test CUDA performance
        x = torch.randn(1000, 1000).cuda()
        start = time.time()
        torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"CUDA computation time: {time.time() - start:.4f} seconds")
except Exception as e:
    print("Error with PyTorch:", e)

# Test OpenCV
print("\n=== Testing OpenCV ===")
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
    
    # Tạo một ảnh test
    img = cv2.imread('test.jpg') if os.path.exists('test.jpg') else \
          cv2.imread('/usr/share/opencv4/samples/data/lena.jpg') if os.path.exists('/usr/share/opencv4/samples/data/lena.jpg') else \
          None
    
    if img is not None:
        print("Image loading: Success")
        print("Image shape:", img.shape)
except Exception as e:
    print("Error with OpenCV:", e)

# Test NumPy
print("\n=== Testing NumPy ===")
try:
    import numpy as np
    print("NumPy version:", np.__version__)
    
    # Test NumPy performance
    arr = np.random.rand(1000, 1000)
    start = time.time()
    np.dot(arr, arr)
    print(f"NumPy computation time: {time.time() - start:.4f} seconds")
except Exception as e:
    print("Error with NumPy:", e)

# Test Matplotlib
print("\n=== Testing Matplotlib ===")
try:
    import matplotlib.pyplot as plt
    print("Matplotlib version:", plt.__version__)
    
    # Tạo plot đơn giản
    plt.figure(figsize=(5,5))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    # plt.show()  # Uncomment nếu muốn hiển thị plot
    plt.close()
    print("Plotting: Success")
except Exception as e:
    print("Error with Matplotlib:", e)

# Test PIL (Pillow)
print("\n=== Testing PIL ===")
try:
    from PIL import Image, __version__
    print("Pillow version:", __version__)
    
    # Tạo ảnh test
    img = Image.new('RGB', (60, 30), color = 'red')
    print("Image creation: Success")
except Exception as e:
    print("Error with PIL:", e)

# Test Tensorboard
print("\n=== Testing Tensorboard ===")
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    print("Tensorboard version:", tensorboard.__version__)
    
    # Test SummaryWriter
    writer = SummaryWriter('runs/test')
    writer.close()
    print("SummaryWriter: Success")
except Exception as e:
    print("Error with Tensorboard:", e)

# Test Seaborn
print("\n=== Testing Seaborn ===")
try:
    import seaborn as sns
    print("Seaborn version:", sns.__version__)
    print("Seaborn: Success")
except Exception as e:
    print("Error with Seaborn:", e)

# System info
print("\n=== System Information ===")
import platform
print("OS:", platform.system(), platform.release())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

if torch.cuda.is_available():
    print("\n=== NVIDIA GPU Information ===")
    try:
        # Thực thi nvidia-smi
        gpu_info = os.popen('nvidia-smi').read()
        print(gpu_info)
    except:
        print("Could not get NVIDIA GPU information")