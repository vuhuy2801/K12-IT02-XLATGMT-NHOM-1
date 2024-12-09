from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from pathlib import Path
import time
from typing import Dict, Any

from src.config import load_config, ModelConfig, PredictionConfig
from src.predict import create_predictor, PredictionException

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(
    title="Animal Detection & Classification API",
    description="API để phát hiện và phân loại động vật",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo predictor global
try:
    _, model_config, _, prediction_config = load_config()
    predictor = create_predictor(model_config, prediction_config)
    logger.info("Đã khởi tạo predictor thành công")
except Exception as e:
    logger.error(f"Lỗi khởi tạo predictor: {str(e)}")
    raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Animal Detection & Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Nhận ảnh và trả về kết quả phát hiện & phân loại
    
    Args:
        file: File ảnh upload
        
    Returns:
        Dict chứa kết quả prediction
    """
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File phải là ảnh (jpg, png, etc.)"
            )
            
        # Đọc và validate ảnh
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Không thể đọc file ảnh: {str(e)}"
            )
            
        # Thực hiện prediction
        try:
            result = predictor.predict(image)
        except PredictionException as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Lỗi không xác định khi predict: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Lỗi server khi xử lý ảnh"
            )
            
        # Thêm thời gian xử lý vào response
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result['processing_time_ms'] = round(processing_time, 2)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Lỗi server không xác định"
        )

@app.get("/health")
async def health_check():
    """Endpoint kiểm tra health của service"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": predictor is not None
    } 