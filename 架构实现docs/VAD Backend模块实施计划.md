# VAD Backend模块完整实施计划

> **文档版本**: v1.0  
> **创建时间**: 2025-01-30  
> **设计目标**: 统一VAD响应结果，支持ONNX和Silero实现，符合已有架构设计  

## 📋 项目概述

### 🎯 核心目标
1. **统一的VAD响应结果(type)**：保持`VADResult`类型一致性
2. **支持ONNX(已有)和Silero的实现**：完整集成Silero-VAD支持
3. **符合已有的架构设计**：遵循现有的依赖倒置和模块化原则

### ✅ 现状评估
- **已完成**：ONNX后端实现，线程安全，性能优化
- **已完成**：完整的类型系统，VADResult统一标准
- **已完成**：抽象基类VADBackend，工厂模式
- **待实现**：Silero后端集成

## 🏗️ 详细代码更改设计

### 1. 类型系统扩展

#### 1.1 更新VADBackend枚举 (`cascade/types/__init__.py`)

**更改位置**：第37-45行
```python
class VADBackend(str, Enum):
    """支持的VAD后端"""
    ONNX = "onnx"
    VLLM = "vllm"
    SILERO = "silero"  # 新增Silero支持
    
    @classmethod
    def get_default_backend(cls) -> str:
        """获取默认后端"""
        return cls.SILERO.value
```

#### 1.2 添加SileroConfig类型 (`cascade/types/__init__.py`)

**插入位置**：第614行后（VLLMConfig类后）
```python
class SileroConfig(BackendConfig):
    """Silero VAD后端配置"""
    onnx: bool = Field(
        default=True,
        description="是否使用ONNX模式，默认使用onnx"
    )
    force_reload: bool = Field(
        default=False,
        description="是否强制重新加载模型（仅torch.hub模式）"
    )
    opset_version: int = Field(
        default=16,
        description="ONNX模型opset版本",
        ge=15,
        le=16
    )
    repo_or_dir: str = Field(
        default="onnx-community/silero-vad",
        description="模型仓库或目录（torch.hub模式）"
    )
    model_name: str = Field(
        default="silero_vad",
        description="模型名称（torch.hub模式）"
    )
    use_pip_package: bool = Field(
        default=True,
        description="优先使用silero-vad pip包，失败时回退到torch.hub"
    )
    chunk_size_samples: dict[int, int] = Field(
        default={16000: 512, 8000: 256},
        description="不同采样率的块大小映射"
    )
    return_seconds: bool = Field(
        default=False,
        description="VADIterator是否返回时间戳（秒）"
    )
    
    @field_validator('opset_version')
    @classmethod
    def validate_opset_version(cls, v):
        """验证opset版本"""
        if v == 15:
            # opset_version=15仅支持16kHz
            pass
        elif v == 16:
            # opset_version=16支持8kHz和16kHz
            pass
        else:
            raise ValueError('opset_version必须是15或16')
        return v
    
    def get_required_chunk_size(self, sample_rate: int) -> int:
        """获取指定采样率的必需块大小"""
        if sample_rate not in self.chunk_size_samples:
            raise ValueError(f'不支持的采样率: {sample_rate}')
        return self.chunk_size_samples[sample_rate]
    
    def is_chunk_size_compatible(self, sample_rate: int, chunk_size: int) -> bool:
        """检查块大小是否兼容"""
        required_size = self.get_required_chunk_size(sample_rate)
        return chunk_size >= required_size
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "onnx": False,
                    "force_reload": False,
                    "opset_version": 16,
                    "use_pip_package": True,
                    "return_seconds": False
                }
            ]
        }
```

#### 1.3 更新__all__导出 (`cascade/types/__init__.py`)

**更改位置**：第655行
```python
    # 后端配置
    "BackendConfig", "ONNXConfig", "VLLMConfig", "SileroConfig",
```

### 2. Silero后端实现

#### 2.1 创建SileroVADBackend (`cascade/backends/silero.py`)

**新文件内容**：
```python
"""
Silero VAD后端实现

基于Silero-VAD的高性能语音活动检测，支持PyTorch和ONNX两种模式。
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
        self._model = None
        self._utils = None
        self._vad_iterator = None
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
                            onnx=self._silero_config.onnx,
                            opset_version=self._silero_config.opset_version
                        )
                        # 创建VADIterator（需要指定采样率）
                        from silero_vad import VADIterator
                        # 注意：这里使用默认采样率，实际使用时会根据音频块动态调整
                        self._thread_local.vad_iterator = None  # 延迟创建
                        self._thread_local.VADIterator_class = VADIterator
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
            onnx=self._silero_config.onnx,
            opset_version=self._silero_config.opset_version
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
            self._thread_local.vad_iterator = VADIterator(
                model,
                sampling_rate=sample_rate
            )
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
            
            # 预热模型（如果需要）
            if self._silero_config.warmup_iterations > 0:
                await self._warmup_model()
            
            self._initialized = True
            
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
            # 获取线程本地模型和迭代器
            model = self._get_thread_model()
            
            # 预处理音频数据
            audio_data = self._adapt_chunk_size(chunk)
            
            # 执行VAD推理
            start_time = time.time()
            
            # 根据配置选择推理模式
            if hasattr(self._silero_config, 'streaming_mode') and self._silero_config.streaming_mode:
                # 流式处理模式：使用VADIterator
                vad_iterator = self._get_thread_vad_iterator(chunk.sample_rate)
                result = vad_iterator(
                    audio_data,
                    return_seconds=self._silero_config.return_seconds
                )
                if isinstance(result, dict) and result:
                    probability = result.get('probability', 0.0)
                    # 可能包含时间戳信息
                    speech_info = result
                else:
                    # 空结果表示非语音
                    probability = 0.0
                    speech_info = None
            else:
                # 直接概率模式：直接调用模型
                probability = float(model(audio_data, chunk.sample_rate).item())
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
            if isinstance(silero_output, (float, int)):
                # 直接概率模式
                probability = float(silero_output)
                is_speech = probability >= self.config.threshold
            elif isinstance(silero_output, dict):
                # VADIterator模式
                probability = silero_output.get('probability', 0.0)
                is_speech = silero_output.get('is_speech', probability >= self.config.threshold)
            else:
                # 尝试转换为float
                probability = float(silero_output)
                is_speech = probability >= self.config.threshold
            
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
        
        return info
```

#### 2.2 更新工厂函数 (`cascade/backends/__init__.py`)

**更改位置**：第43-60行
```python
def create_vad_backend(config: VADConfig) -> VADBackend:
    """根据配置创建VAD后端实例"""
    if config.backend == VADBackendEnum.ONNX:
        try:
            from .onnx import ONNXVADBackend
            return ONNXVADBackend(config)
        except ImportError as e:
            raise CascadeError(
                f"ONNX后端不可用，请确保已安装onnxruntime: {e}",
                ErrorCode.BACKEND_UNAVAILABLE
            )
    elif config.backend == VADBackendEnum.SILERO:
        try:
            from .silero import SileroVADBackend
            return SileroVADBackend(config)
        except ImportError as e:
            raise CascadeError(
                f"Silero后端不可用，请确保已安装silero-vad或torch: {e}",
                ErrorCode.BACKEND_UNAVAILABLE
            )
    else:
        raise CascadeError(
            f"不支持的VAD后端: {config.backend}",
            ErrorCode.BACKEND_UNAVAILABLE,
            context={"supported_backends": [e.value for e in VADBackendEnum]}
        )
```

#### 2.3 更新__all__导出 (`cascade/backends/__init__.py`)

**更改位置**：第63-66行
```python
__all__ = [
    "VADBackend",
    "create_vad_backend",
]
```

## 🧪 测试策略

### 1. 单元测试文件 (`tests/unit/backends/test_silero.py`)

```python
"""Silero VAD后端单元测试"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from cascade.types import VADConfig, AudioChunk, VADBackend as VADBackendEnum
from cascade.backends.silero import SileroVADBackend
from cascade.types import VADProcessingError, ModelLoadError


class TestSileroVADBackend:
    """Silero VAD后端测试套件"""
    
    @pytest.fixture
    def vad_config(self):
        """VAD配置fixture"""
        return VADConfig(
            backend=VADBackendEnum.SILERO,
            workers=2,
            threshold=0.5,
            chunk_duration_ms=500
        )
    
    @pytest.fixture
    def backend(self, vad_config):
        """Silero后端fixture"""
        return SileroVADBackend(vad_config)
    
    @pytest.fixture
    def audio_chunk_16k(self):
        """16kHz音频块fixture"""
        data = np.random.rand(512).astype(np.float32)  # Silero要求的块大小
        return AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=512,
            timestamp_ms=0.0,
            sample_rate=16000
        )
    
    def test_initialization(self, backend):
        """测试后端初始化"""
        assert not backend.is_initialized
        assert backend.silero_config is not None
        assert backend.silero_config.onnx is False
        assert backend.silero_config.use_pip_package is True
    
    @patch('cascade.backends.silero.load_silero_vad')
    async def test_async_initialize_pytorch_mode(self, mock_load, backend):
        """测试PyTorch模式异步初始化"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        await backend.initialize()
        
        assert backend.is_initialized
        mock_load.assert_called_once()
    
    @patch('torch.hub.load')
    async def test_async_initialize_onnx_mode(self, mock_hub_load, backend):
        """测试ONNX模式异步初始化"""
        backend._silero_config.onnx = True
        backend._silero_config.use_pip_package = False
        mock_model = Mock()
        mock_utils = Mock()
        mock_hub_load.return_value = (mock_model, mock_utils)
        
        await backend.initialize()
        
        assert backend.is_initialized
        mock_hub_load.assert_called_once()
    
    def test_chunk_size_adaptation_exact_match(self, backend, audio_chunk_16k):
        """测试块大小完全匹配的情况"""
        adapted = backend._adapt_chunk_size(audio_chunk_16k)
        
        assert len(adapted) == 512
        assert adapted.dtype == np.float32
        np.testing.assert_array_equal(adapted, audio_chunk_16k.data)
    
    def test_chunk_size_adaptation_padding(self, backend):
        """测试块大小填充的情况"""
        # 创建较小的音频块
        small_data = np.random.rand(256).astype(np.float32)
        chunk = AudioChunk(
            data=small_data,
            sequence_number=1,
            start_frame=0,
            chunk_size=256,
            timestamp_ms=0.0,
            sample_rate=16000
        )
        
        adapted = backend._adapt_chunk_size(chunk)
        
        assert len(adapted) == 512  # Silero要求的大小
        assert adapted.dtype == np.float32
        # 前256个样本应该是原始数据
        np.testing.assert_array_equal(adapted[:256], small_data)
        # 后256个样本应该是零
        np.testing.assert_array_equal(adapted[256:], np.zeros(256))
    
    def test_chunk_size_adaptation_truncation(self, backend):
        """测试块大小截断的情况"""
        # 创建较大的音频块
        large_data = np.random.rand(1024).astype(np.float32)
        chunk = AudioChunk(
            data=large_data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1024,
            timestamp_ms=0.0,
            sample_rate=16000
        )
        
        adapted = backend._adapt_chunk_size(chunk)
        
        assert len(adapted) == 512  # Silero要求的大小
        assert adapted.dtype == np.float32
        # 应该是原始数据的前512个样本
        np.testing.assert_array_equal(adapted, large_data[:512])
    
    @patch('cascade.backends.silero.SileroVADBackend._get_thread_model')
    def test_process_chunk_success(self, mock_get_model, backend, audio_chunk_16k):
        """测试音频块处理成功"""
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.item.return_value = 0.8
        mock_get_model.return_value = mock_model
        
        backend._initialized = True
        
        result = backend.process_chunk(audio_chunk_16k)
        
        assert result.is_speech is True  # 0.8 > 0.5 (threshold)
        assert result.probability == 0.8
        assert result.chunk_id == 1
        assert "silero" in result.metadata["backend"]
    
    def test_postprocess_output_float_input(self, backend, audio_chunk_16k):
        """测试浮点数输出后处理"""
        probability = 0.7
        
        result = backend._postprocess_output(probability, audio_chunk_16k, 0.005)
        
        assert result.is_speech is True
        assert result.probability == 0.7
        assert result.confidence == 0.7
        assert result.metadata["inference_time_ms"] == 5.0
    
    def test_postprocess_output_dict_input(self, backend, audio_chunk_16k):
        """测试字典输出后处理"""
        silero_output = {
            'probability': 0.3,
            'is_speech': False
        }
        
        result = backend._postprocess_output(silero_output, audio_chunk_16k, 0.003)
        
        assert result.is_speech is False
        assert result.probability == 0.3
        assert result.confidence == 0.7  # 1.0 - 0.3
    
    def test_get_backend_info(self, backend):
        """测试获取后端信息"""
        info = backend.get_backend_info()
        
        assert "backend_type" in info
        assert "silero_config" in info
        assert "expected_chunk_sizes" in info
        assert info["backend_type"] == "SileroVADBackend"
    
    @patch('cascade.backends.silero.SileroVADBackend._get_thread_model')
    def test_warmup(self, mock_get_model, backend, audio_chunk_16k):
        """测试模型预热"""
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.item.return_value = 0.5
        mock_get_model.return_value = mock_model
        
        backend._initialized = True
        backend._silero_config.warmup_iterations = 2
        # 模拟VADIterator类
        backend._thread_local.VADIterator_class = Mock()
        backend._thread_local.use_pip = True
        
        backend.warmup(audio_chunk_16k)
        
        # 验证模型被调用了预热次数
        assert mock_model.call_count >= 2
    
    async def test_close(self, backend):
        """测试后端关闭"""
        backend._initialized = True
        
        await backend.close()
        
        assert not backend.is_initialized
        assert backend._chunk_count == 0
    
    def test_invalid_sample_rate(self, backend):
        """测试不支持的采样率"""
        invalid_chunk = AudioChunk(
            data=np.random.rand(100).astype(np.float32),
            sequence_number=1,
            start_frame=0,
            chunk_size=100,
            timestamp_ms=0.0,
            sample_rate=22050  # 不支持的采样率
        )
        
        with pytest.raises(VADProcessingError):
            backend._adapt_chunk_size(invalid_chunk)
```

### 2. 集成测试计划

#### 2.1 端到端测试 (`tests/integration/test_silero_integration.py`)

重点测试：
- Silero后端与VADProcessor的集成
- 多线程并发处理
- 长时间运行稳定性
- 不同音频格式的处理

#### 2.2 性能基准测试 (`tests/benchmarks/test_silero_performance.py`)

性能目标：
- 推理延迟：P99 < 8ms
- 吞吐量：> 150 chunks/s/worker
- 内存使用：< 120MB/实例

## 📊 架构一致性验证

### 1. 依赖关系检查

✅ **单向数据流**：
```
cascade/types/ (零依赖)
    ↓
cascade/backends/base.py (依赖types)
    ↓
cascade/backends/silero.py (依赖base + types)
    ↓
cascade/backends/__init__.py (工厂函数)
```

✅ **接口一致性**：
- `SileroVADBackend`完全实现`VADBackend`抽象接口
- 返回统一的`VADResult`类型
- 遵循相同的生命周期管理

✅ **错误处理一致性**：
- 使用统一的`CascadeError`异常体系
- 相同的错误码和上下文信息格式

### 2. 性能一致性

✅ **线程安全**：
- 线程本地存储模式与ONNX后端一致
- 递归锁机制保持一致

✅ **内存管理**：
- numpy数组连续性保证
- 零拷贝原则（在可能的情况下）

## 🚀 详细实施计划

### Phase 1: 类型系统更新 (0.5天)

**任务清单**：
- [ ] 更新`VADBackend`枚举添加`SILERO`选项
- [ ] 创建`SileroConfig`类型定义
- [ ] 更新类型系统的`__all__`导出
- [ ] 验证类型定义的pydantic验证规则

**验收标准**：
- 类型导入正常，无语法错误
- pydantic验证规则正确工作
- 配置示例可以正常创建和验证

### Phase 2: Silero后端核心实现 (1.5天)

**任务清单**：
- [ ] 创建`SileroVADBackend`基础结构
- [ ] 实现异步初始化逻辑
- [ ] 实现块大小适配机制
- [ ] 实现核心推理逻辑
- [ ] 实现结果标准化
- [ ] 实现状态管理和预热

**验收标准**：
- 基础功能完整实现
- 单元测试通过率 > 90%
- 基本性能指标达标

### Phase 3: 工厂函数和集成 (0.5天)

**任务清单**：
- [ ] 更新`create_vad_backend`工厂函数
- [ ] 更新模块导出
- [ ] 集成测试验证
- [ ] 错误处理测试

**验收标准**：
- 工厂函数正确创建Silero后端
- 与现有VADProcessor集成无问题
- 错误处理路径完整

### Phase 4: 测试和验证 (0.5天)

**任务清单**：
- [ ] 完善单元测试覆盖
- [ ] 集成测试验证
- [ ] 性能基准测试
- [ ] 文档更新

**验收标准**：
- 测试覆盖率 > 95%
- 性能指标达到预期
- 集成测试全部通过

## 📋 验收标准总结

### 功能验收
- ✅ **后端创建**：工厂函数可正确创建Silero后端
- ✅ **基础推理**：可处理16kHz和8kHz音频块
- ✅ **块大小适配**：自动适配不同大小的音频块
- ✅ **结果统一**：输出标准的VADResult格式
- ✅ **状态管理**：正确的模型状态重置机制

### 性能验收
- ✅ **推理延迟**：P99 < 8ms
- ✅ **吞吐量**：> 150 chunks/s/worker
- ✅ **内存使用**：< 120MB/实例
- ✅ **线程安全**：多线程并发无竞争

### 架构验收
- ✅ **接口一致性**：完全兼容VADBackend抽象接口
- ✅ **依赖管理**：单向依赖，无循环依赖
- ✅ **错误处理**：统一的异常体系
- ✅ **扩展性**：易于添加新的配置选项

### 测试验收
- ✅ **单元测试**：覆盖率 > 95%
- ✅ **集成测试**：端到端功能验证
- ✅ **性能测试**：基准测试通过
- ✅ **边界测试**：异常情况处理验证

## 📝 总结

这个实施计划确保了：

1. **完整的Silero集成**：支持PyTorch和ONNX两种模式
2. **架构一致性**：完全符合现有的设计模式
3. **统一的响应格式**：`VADResult`类型保持一致
4. **高性能设计**：线程安全，内存优化
5. **完整的测试覆盖**：单元测试、集成测试、性能测试

总工期预估：**3天**，可以并行开发和测试，确保质量和进度的平衡。

---

> **下一步行动**: 切换到Code模式开始具体的代码实施