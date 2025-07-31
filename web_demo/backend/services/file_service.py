import os
import uuid
import logging
import aiofiles
from typing import Dict, Any, Optional
import mimetypes

from .vad_service import vad_service

logger = logging.getLogger(__name__)

# 上传目录
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../uploads'))

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

class FileService:
    """文件处理服务"""

    @staticmethod
    async def save_uploaded_file(file_content: bytes, filename: str) -> str:
        """保存上传的文件"""
        # 生成唯一文件ID
        file_id = str(uuid.uuid4())

        # 获取文件扩展名
        _, ext = os.path.splitext(filename)

        # 保存文件
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        logger.info(f"文件已保存: {file_path}")

        return file_path

    @staticmethod
    async def process_audio_file(file_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理音频文件"""
        try:
            # 获取文件名
            filename = os.path.basename(file_path)

            # 处理音频文件
            result = await vad_service.process_file(file_path, config)

            # 添加文件信息
            result['file_id'] = os.path.splitext(filename)[0]
            result['filename'] = filename
            result['status'] = 'success'

            return result

        except Exception as e:
            logger.error(f"处理音频文件失败: {e}")
            return {
                'status': 'error',
                'file_id': os.path.splitext(os.path.basename(file_path))[0],
                'filename': os.path.basename(file_path),
                'error': str(e)
            }

    @staticmethod
    def validate_audio_file(content_type: str, file_size: int) -> Dict[str, Any]:
        """验证音频文件"""
        # 检查文件类型
        valid_types = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg']
        if content_type not in valid_types:
            return {
                'valid': False,
                'error': 'unsupported_format',
                'message': f"不支持的文件格式: {content_type}",
                'details': {
                    'format': content_type,
                    'supported_formats': valid_types
                }
            }

        # 检查文件大小
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return {
                'valid': False,
                'error': 'file_too_large',
                'message': f"文件过大: {file_size} 字节 (最大 {max_size} 字节)",
                'details': {
                    'size': file_size,
                    'max_size': max_size
                }
            }

        return {'valid': True}

# 创建全局文件服务实例
file_service = FileService()