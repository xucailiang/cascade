"""
Silero VAD后端实现

基于Silero-VAD的高性能语音活动检测，支持PyTorch和ONNX两种模式。
遵循silero-vad的实际API使用方式。
"""

import threading
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from cascade.types import (
    AudioChunk,
    VADResult,
    VADConfig,
    SileroConfig,
    CascadeError,
    ErrorCode,
    ModelLoadError,
    VADProcessingError
)
from .base import VADBackend


class SileroVADBackend(VADBackend):
    """
    Silero VAD后端实现
    
    特点：
    - 支持PyTorch和ONNX两种推理模式
    - 自动块大小适配和填充策略
    - 线程本地模型实例管理
    - 完整的状态重置机制
    - 流式和批量处理支持
    
    遵循silero-vad的实际API使用方式：
    - pip package优先，torch.hub回退
    - 正确的VADIterator创建和使用
    - 准确的模型状态重置
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
        self._thread_local = threading.local()
        self._expected_chunk_sizes = {}
        self._chunk_count = 0
        self._reset_interval = 1000  # 每1000块重置一次状态
        
    @property
    def silero_config(self) -> SileroConfig:
        """获取Silero配置"""
        return self._silero_config
    
    def _get_thread_model(self):
        """
        获取线程本地的Silero模型实例
        
        使用线程本地存储确保每个线程都有独立的模型实例，
        避免并发访问冲突和状态混乱。
        
        Returns:
            Silero VAD模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时
        """
        if not hasattr(self._thread_local, 'model'):
            try:
                if self._silero_config.use_pip_package:
                    # 优先使用pip package
                    try:
                        from silero_vad import load_silero_vad
                        self._thread_local.model = load_silero_vad(
                            onnx=self._silero_config.onnx
                        )
                        # 创建VADIterator类引用（延迟创建实例）
                        from silero_vad import VADIterator
                        self._thread_local.VADIterator_class = VADIterator
                        self._thread_local.vad_iterator = None  # 延迟创建
                        self._thread_local.use_pip = True
                    except ImportError:
                        # 回退到torch.hub
                        self._load_via_torch_hub()
                else:
                    # 直接使用torch.hub
                    self._load_via_torch_hub()
            
            except Exception as e:
                raise ModelLoadError(
                    "silero_vad",
                    f"创建Silero模型失败: {str(e)}"
                )
        
        return self._thread_local.model
    
    def _load_via_torch_hub(self):
        """通过torch.hub加载模型"""
        import torch
        self._thread_local.model, self._thread_local.utils = torch.hub.load(
            repo_or_dir=self._silero_config.repo_or_dir,
            model=self._silero_config.model_name,
            force_reload=self._silero_config.force_reload,
            onnx=self._silero_config.onnx
        )
        
        # 解包工具函数
        (get_speech_timestamps, save_audio, read_audio, 
         VADIterator, collect_chunks) = self._thread_local.utils
        
        self._thread_local.VADIterator_class = VADIterator
        self._thread_local.vad_iterator = None  # 延迟创建
        self._thread_local.use_pip = False
    
    def _get_thread_vad_iterator(self, sample_rate: int):
        """获取指定采样率的线程本地VAD迭代器"""
        # 确保模型已加载
        model = self._get_thread_model()
        
        # 检查是否需要创建新的迭代器（采样率变化时）
        if (not hasattr(self._thread_local, 'vad_iterator') or 
            self._thread_local.vad_iterator is None or
            getattr(self._thread_local, 'current_sample_rate', None) != sample_rate):
            
            VADIterator = self._thread_local.VADIterator_class
            print(f"[DEBUG] 创建VADIterator，采样率: {sample_rate}, threshold: {self.config.threshold}")
            self._thread_local.vad_iterator = VADIterator(
                model,
                sampling_rate=sample_rate,
                threshold=self.config.threshold  # 确保传递threshold参数
            )
            print(f"[DEBUG] VADIterator创建完成")
            self._thread_local.current_sample_rate = sample_rate
        
        return self._thread_local.vad_iterator
    
    async def initialize(self) -> None:
        """
        异步初始化Silero后端
        
        检查依赖可用性并创建测试模型实例。
        
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
                        )
            else:
                import torch
                try:
                    torch.hub.list(self._silero_config.repo_or_dir)
                except Exception as e:
                    raise CascadeError(
                        f"无法访问Silero模型仓库: {e}",
                        ErrorCode.BACKEND_UNAVAILABLE
                    )
            
            # 预计算期望的块大小
            for sample_rate in [8000, 16000]:
                if sample_rate in self._silero_config.chunk_size_samples:
                    self._expected_chunk_sizes[sample_rate] = (
                        self._silero_config.chunk_size_samples[sample_rate]
                    )
            
            # 创建测试模型实例验证可用性
            test_model = self._get_thread_model()
            
            self._initialized = True
            
            # 预热模型（如果需要）
            if self._silero_config.warmup_iterations > 0:
                await self._warmup_model()
            
        except ImportError as e:
            raise CascadeError(
                f"Silero VAD依赖不可用: {e}。请安装: pip install silero-vad 或确保torch可用",
                ErrorCode.BACKEND_UNAVAILABLE
            )
        except Exception as e:
            if isinstance(e, (CascadeError, ModelLoadError)):
                raise
            raise ModelLoadError(
                "silero_vad",
                f"初始化失败: {str(e)}"
            )
    
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块并返回VAD检测结果
        
        自动适配块大小并执行VAD推理。
        
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
        
        try:
            # 获取线程本地模型
            model = self._get_thread_model()
            
            # 预处理音频数据
            audio_data = self._adapt_chunk_size(chunk)
            
            # 执行VAD推理
            start_time = time.time()
            
            # 将numpy数组转换为PyTorch tensor（silero-vad需要torch tensor）
            import torch
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data.copy())
            else:
                audio_tensor = audio_data
            
            # 根据配置选择推理模式
            if hasattr(self._silero_config, 'streaming_mode') and self._silero_config.streaming_mode:
                # 流式处理模式：使用VADIterator
                print(f"使用silero-vad的VADIterator进行流式处理，采样率: {chunk.sample_rate}")
                vad_iterator = self._get_thread_vad_iterator(chunk.sample_rate)
                result = vad_iterator(
                    audio_tensor,
                    return_seconds=self._silero_config.return_seconds
                )
                # result: {'start': 11808} 表示语音开始的时间戳
                # result: {'end': 82400} 表示语音结束的时间戳
                
                print(f"silero-vad的原始推理结果:{result}")
                
                # 保存原始结果到线程本地存储，供测试脚本使用
                self._thread_local.last_vad_result = result
                
                if isinstance(result, dict) and result:
                    # VADIterator返回语音段边界信息
                    if 'start' in result or 'end' in result:
                        # 边界事件，概率设为1.0表示检测到边界
                        probability = 1.0
                    else:
                        probability = result.get('probability', 0.0)
                    speech_info = result
                else:
                    # 空结果表示当前块无语音边界变化
                    probability = 0.0
                    speech_info = None
            else:
                # 直接概率模式：直接调用模型
                probability = float(model(audio_tensor, chunk.sample_rate).item())
                speech_info = None
            
            inference_time = time.time() - start_time
            
            # 后处理结果
            output_data = speech_info if speech_info else probability
            vad_result = self._postprocess_output(
                output_data,
                chunk,
                inference_time
            )
            
            # 定期重置状态防止状态累积
            self._chunk_count += 1
            if self._chunk_count % self._reset_interval == 0:
                self._reset_model_states()
            
            return vad_result
            
        except Exception as e:
            raise VADProcessingError(
                f"Silero推理失败: {str(e)}",
                ErrorCode.INFERENCE_FAILED,
                context={
                    "chunk_id": chunk.sequence_number,
                    "chunk_size": chunk.chunk_size,
                    "sample_rate": chunk.sample_rate,
                    "silero_mode": "onnx" if self._silero_config.onnx else "pytorch"
                }
            )
    
    def warmup(self, dummy_chunk: AudioChunk) -> None:
        """
        使用虚拟数据预热模型
        
        消除首次推理的冷启动延迟。
        
        Args:
            dummy_chunk: 用于预热的虚拟音频块
            
        Raises:
            VADProcessingError: 当预热失败时
        """
        try:
            for _ in range(self._silero_config.warmup_iterations):
                _ = self.process_chunk(dummy_chunk)
            
            # 重置状态确保预热不影响实际处理
            self._reset_model_states()
            self._chunk_count = 0
            
        except Exception as e:
            raise VADProcessingError(
                f"Silero模型预热失败: {str(e)}",
                ErrorCode.INFERENCE_FAILED
            )
    
    async def close(self) -> None:
        """
        异步关闭后端并释放资源
        
        清理模型实例和相关资源。
        """
        try:
            # 清理线程本地资源
            if hasattr(self._thread_local, 'model'):
                delattr(self._thread_local, 'model')
            if hasattr(self._thread_local, 'vad_iterator'):
                delattr(self._thread_local, 'vad_iterator')
            if hasattr(self._thread_local, 'utils'):
                delattr(self._thread_local, 'utils')
            
            self._initialized = False
            self._chunk_count = 0
            
        except Exception:
            # 静默处理清理错误
            pass
    
    def _adapt_chunk_size(self, chunk: AudioChunk) -> np.ndarray:
        """
        适配块大小到Silero要求
        
        Silero要求固定的块大小：16kHz=512样本，8kHz=256样本
        
        策略：
        1. 如果输入块大小匹配Silero要求，直接使用
        2. 如果输入块更大，取前N个样本
        3. 如果输入块更小，零填充到要求大小
        
        Args:
            chunk: 输入音频块
            
        Returns:
            适配后的音频数据
        """
        try:
            required_size = self._silero_config.get_required_chunk_size(chunk.sample_rate)
            audio_data = np.asarray(chunk.data, dtype=np.float32)
            
            # 确保是一维数组
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            
            current_size = len(audio_data)
            
            if current_size == required_size:
                # 大小匹配，直接使用
                return audio_data
            elif current_size < required_size:
                # 零填充
                padded = np.zeros(required_size, dtype=np.float32)
                padded[:current_size] = audio_data
                return padded
            else:
                # 截取前required_size个样本
                return audio_data[:required_size]
                
        except Exception as e:
            raise VADProcessingError(
                f"块大小适配失败: {str(e)}",
                ErrorCode.INVALID_INPUT,
                context={
                    "input_size": len(chunk.data) if hasattr(chunk.data, '__len__') else 'unknown',
                    "required_size": self._silero_config.get_required_chunk_size(chunk.sample_rate),
                    "sample_rate": chunk.sample_rate
                }
            )
    
    def _postprocess_output(
        self,
        silero_output: Union[float, dict],
        chunk: AudioChunk,
        inference_time: float
    ) -> VADResult:
        """
        将Silero输出标准化为VADResult
        
        支持多种Silero输出格式：
        1. 直接概率值 (float)
        2. VADIterator输出 (dict)
        
        Args:
            silero_output: Silero模型输出
            chunk: 输入音频块
            inference_time: 推理耗时
            
        Returns:
            标准化的VAD结果
        """
        try:
            print(f"[DEBUG] 当前threshold配置: {self.config.threshold}")
            print(f"[DEBUG] Silero输出类型: {type(silero_output)}, 值: {silero_output}")
            
            if isinstance(silero_output, (float, int)):
                # 直接概率模式
                probability = float(silero_output)
                is_speech = probability >= self.config.threshold
                print(f"[DEBUG] 直接概率模式 - 概率: {probability}, threshold: {self.config.threshold}, is_speech: {is_speech}")
            elif isinstance(silero_output, dict):
                # VADIterator模式
                probability = silero_output.get('probability', 0.0)
                is_speech = silero_output.get('is_speech', probability >= self.config.threshold)
                print(f"[DEBUG] VADIterator模式 - 概率: {probability}, threshold: {self.config.threshold}, is_speech: {is_speech}")
            else:
                # 尝试转换为float
                probability = float(silero_output)
                is_speech = probability >= self.config.threshold
                print(f"[DEBUG] 其他模式 - 概率: {probability}, threshold: {self.config.threshold}, is_speech: {is_speech}")
            
            # 计算置信度
            confidence = probability if is_speech else (1.0 - probability)
            
            # 检查是否进行了块大小适配
            required_size = self._silero_config.get_required_chunk_size(chunk.sample_rate)
            chunk_adapted = len(chunk.data) != required_size
            
            return VADResult(
                is_speech=is_speech,
                probability=probability,
                start_ms=chunk.timestamp_ms,
                end_ms=chunk.get_end_timestamp_ms(),
                chunk_id=chunk.sequence_number,
                confidence=confidence,
                metadata={
                    "inference_time_ms": inference_time * 1000,
                    "backend": "silero",
                    "mode": "onnx" if self._silero_config.onnx else "pytorch",
                    "chunk_adapted": chunk_adapted,
                    "required_chunk_size": required_size,
                    "actual_chunk_size": len(chunk.data),
                    "streaming_mode": getattr(self._silero_config, 'streaming_mode', False),
                    "model_repo": self._silero_config.repo_or_dir
                }
            )
            
        except Exception as e:
            raise VADProcessingError(
                f"Silero输出后处理失败: {str(e)}",
                ErrorCode.RESULT_VALIDATION_FAILED,
                context={
                    "output_type": type(silero_output).__name__,
                    "output_value": str(silero_output)[:100]  # 限制长度
                }
            )
    
    def _reset_model_states(self) -> None:
        """
        重置模型状态
        
        在以下情况调用：
        1. 预热完成后
        2. 定期重置防止状态累积
        3. 新音频流开始时
        """
        try:
            # 重置模型状态
            model = self._get_thread_model()
            if hasattr(model, 'reset_states'):
                model.reset_states()
            
            # 重置VAD迭代器状态（如果存在）
            if hasattr(self._thread_local, 'vad_iterator') and self._thread_local.vad_iterator:
                if hasattr(self._thread_local.vad_iterator, 'reset_states'):
                    self._thread_local.vad_iterator.reset_states()
                
        except Exception:
            # 静默处理重置错误，不影响主要功能
            pass
    
    async def _warmup_model(self) -> None:
        """异步预热模型"""
        try:
            # 创建虚拟音频块进行预热
            for sample_rate in [8000, 16000]:
                if sample_rate in self._expected_chunk_sizes:
                    chunk_size = self._expected_chunk_sizes[sample_rate]
                    dummy_data = np.zeros(chunk_size, dtype=np.float32)
                    
                    dummy_chunk = AudioChunk(
                        data=dummy_data,
                        sequence_number=0,
                        start_frame=0,
                        chunk_size=chunk_size,
                        timestamp_ms=0.0,
                        sample_rate=sample_rate
                    )
                    
                    # 执行预热推理
                    for _ in range(self._silero_config.warmup_iterations):
                        self.process_chunk(dummy_chunk)
            
            # 重置状态
            self._reset_model_states()
            self._chunk_count = 0
            
        except Exception as e:
            raise ModelLoadError(
                "silero_vad",
                f"模型预热失败: {str(e)}"
            )
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        获取Silero后端详细信息
        
        Returns:
            包含后端信息的字典
        """
        info = super().get_backend_info()
        info.update({
            "silero_config": self._silero_config.__dict__,
            "expected_chunk_sizes": self._expected_chunk_sizes,
            "chunk_count": self._chunk_count,
            "reset_interval": self._reset_interval
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