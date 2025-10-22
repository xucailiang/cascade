"""
Silero VAD后端实现 - 简化版

基于1:1:1:1架构优化，移除线程本地存储和复杂的模型管理。
每个实例拥有独立的模型，由调用者（StreamProcessor）直接管理。
"""

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from cascade.types import (
    AudioChunk,
    CascadeError,
    ErrorCode,
    ModelLoadError,
    SileroConfig,
    VADConfig,
    VADProcessingError,
    VADResult,
)

from .base import VADBackend

logger = logging.getLogger(__name__)


class SileroVADBackend(VADBackend):
    """
    Silero VAD后端实现 - 简化版
    
    简化设计原则：
    - 移除threading.local()，每个实例独立
    - 移除复杂的线程本地模型管理
    - 简化为直接的模型加载和封装
    - 保留异步接口便于扩展
    
    注意：此类现在主要用于可选的后端抽象层。
    StreamProcessor可以直接使用silero-vad，无需此类。
    """

    def __init__(self, vad_config: VADConfig):
        """
        初始化Silero VAD后端
        
        Args:
            vad_config: VAD配置对象
        """
        super().__init__(vad_config)
        
        # 创建默认Silero配置
        self._silero_config = SileroConfig()
        
        # 模型和VADIterator（简化：直接存储）
        self.model = None
        self.vad_iterator = None
    
    @property
    def silero_config(self) -> SileroConfig:
        """获取Silero配置"""
        return self._silero_config
    
    async def initialize(self) -> None:
        """
        异步初始化Silero后端
        
        加载模型和创建VADIterator。
        
        Raises:
            CascadeError: 当依赖不可用时
            ModelLoadError: 当模型加载失败时
        """
        try:
            # 检查依赖可用性
            if self._silero_config.use_pip_package:
                try:
                    import silero_vad
                except ImportError:
                    # 检查torch.hub作为后备
                    import torch
                    try:
                        torch.hub.list(self._silero_config.repo_or_dir)
                    except Exception as e:
                        raise CascadeError(
                            f"silero-vad pip包不可用且torch.hub也无法访问: {e}",
                            ErrorCode.BACKEND_UNAVAILABLE
                        ) from e
            else:
                import torch
                try:
                    torch.hub.list(self._silero_config.repo_or_dir)
                except Exception as e:
                    raise CascadeError(
                        f"无法访问Silero模型仓库: {e}",
                        ErrorCode.BACKEND_UNAVAILABLE
                    ) from e
            
            # 加载模型
            from silero_vad import load_silero_vad, VADIterator
            
            logger.info("开始加载Silero VAD模型...")
            self.model = await asyncio.to_thread(
                load_silero_vad,
                onnx=self._silero_config.onnx
            )
            
            # 创建VADIterator
            self.vad_iterator = VADIterator(
                self.model,
                sampling_rate=16000,
                threshold=self.config.threshold,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms
            )
            
            self._initialized = True
            logger.info("Silero VAD后端初始化完成")
            
        except ImportError as e:
            raise CascadeError(
                f"Silero VAD依赖不可用: {e}。请安装: pip install silero-vad 或确保torch可用",
                ErrorCode.BACKEND_UNAVAILABLE
            ) from e
        except Exception as e:
            if isinstance(e, (CascadeError, ModelLoadError)):
                raise
            raise ModelLoadError(
                "silero_vad",
                f"初始化失败: {str(e)}"
            ) from e
    
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块并返回VAD检测结果
        
        Args:
            chunk: 音频数据块
            
        Returns:
            VAD检测结果
            
        Raises:
            VADProcessingError: 当推理失败时
        """
        # 确保初始化和输入验证
        self._ensure_initialized()
        self._validate_chunk(chunk)
        
        if self.vad_iterator is None:
            raise VADProcessingError(
                "VAD iterator未初始化",
                ErrorCode.INVALID_STATE
            )
        
        try:
            # 准备音频数据
            if isinstance(chunk.data, np.ndarray):
                audio_tensor = torch.from_numpy(chunk.data.copy())
            else:
                audio_tensor = chunk.data
            
            # 执行VAD推理
            result = self.vad_iterator(
                audio_tensor,
                return_seconds=self._silero_config.return_seconds
            )
            
            # 标准化输出
            if result:
                if 'start' in result or 'end' in result:
                    probability = 1.0
                else:
                    probability = result.get('probability', 0.0)
                speech_info = result
            else:
                probability = 0.0
                speech_info = None
            
            # 构建VADResult
            start_ms = chunk.timestamp_ms
            end_ms = start_ms + (chunk.chunk_size / chunk.sample_rate * 1000)
            
            return VADResult(
                audio_chunk=chunk.data,
                original_result=speech_info,
                is_speech=probability >= self.config.threshold,
                probability=probability,
                chunk_id=chunk.sequence_number,
                start_ms=start_ms,
                end_ms=end_ms,
                metadata={
                    "backend": "silero",
                    "mode": "onnx" if self._silero_config.onnx else "pytorch",
                }
            )
            
        except Exception as e:
            raise VADProcessingError(
                f"Silero推理失败: {str(e)}",
                ErrorCode.INFERENCE_FAILED,
                context={
                    "chunk_id": chunk.sequence_number,
                    "chunk_size": chunk.chunk_size,
                    "sample_rate": chunk.sample_rate,
                }
            ) from e
    
    async def close(self) -> None:
        """
        异步关闭后端并释放资源
        """
        try:
            if self.vad_iterator:
                self.vad_iterator.reset_states()
            
            # 清理资源
            self.model = None
            self.vad_iterator = None
            self._initialized = False
            
        except Exception:
            # 静默处理清理错误
            pass
    
    def get_backend_info(self) -> dict[str, Any]:
        """
        获取Silero后端详细信息
        
        Returns:
            包含后端信息的字典
        """
        info = super().get_backend_info()
        info.update({
            "silero_config": self._silero_config.__dict__,
        })
        
        # 检查依赖可用性
        try:
            if self._silero_config.use_pip_package:
                try:
                    import silero_vad
                    info["silero_vad_version"] = getattr(silero_vad, '__version__', 'unknown')
                    info["source"] = "pip_package"
                except ImportError:
                    import torch
                    info["torch_version"] = torch.__version__
                    info["silero_vad_version"] = "torch.hub"
                    info["source"] = "torch_hub"
            else:
                import torch
                info["torch_version"] = torch.__version__
                info["silero_vad_version"] = "torch.hub"
                info["source"] = "torch_hub"
            
            info["onnx_mode"] = self._silero_config.onnx
        except ImportError:
            info["silero_vad_version"] = "not_installed"
            info["source"] = "unavailable"
        
        return info
