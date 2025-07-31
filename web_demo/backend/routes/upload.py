import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from ..services.file_service import file_service
from ..models import FileUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_audio_file(
    file: UploadFile = File(...),
    vad_threshold: Optional[float] = Form(0.5),
    chunk_duration_ms: Optional[int] = Form(512),
    overlap_ms: Optional[int] = Form(32),
    workers: Optional[int] = Form(4),
    backend: Optional[str] = Form("silero")
):
    """处理音频文件上传"""
    
    validation = file_service.validate_audio_file(file.content_type, file.size)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation)
    
    try:
        file_content = await file.read()
        file_path = await file_service.save_uploaded_file(file_content, file.filename)
        
        config = {
            'threshold': vad_threshold,
            'chunk_duration_ms': chunk_duration_ms,
            'overlap_ms': overlap_ms,
            'workers': workers,
            'backend': backend
        }
        
        result = await file_service.process_audio_file(file_path, config)
        
        # 将numpy数组转换为列表以进行JSON序列化
        if 'audio_data' in result and isinstance(result['audio_data'], np.ndarray):
            result['audio_data'] = result['audio_data'].tolist()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"文件上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))