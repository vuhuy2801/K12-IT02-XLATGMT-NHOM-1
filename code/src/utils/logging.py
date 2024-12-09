import logging
import sys
import codecs
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Fix encoding cho Windows
if sys.platform.startswith('win'):
    # Chỉ set encoding cho stdout nếu nó chưa được wrapped
    if not isinstance(sys.stdout, codecs.StreamWriter):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    if not isinstance(sys.stderr, codecs.StreamWriter):    
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def setup_logging(log_dir: str = "logs") -> None:
    """Cấu hình logging system"""
    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler với encoding UTF-8
    file_handler = RotatingFileHandler(
        log_dir / "animal_classifier.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Thêm encoding UTF-8
    )
    file_handler.setFormatter(formatter)
    
    # Console handler với encoding UTF-8 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Xóa handlers cũ nếu có
    root_logger.handlers.clear()
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logging from libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING) 