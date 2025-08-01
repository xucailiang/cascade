"""
ONNX VAD后端实现

基于ONNXRuntime的高性能VAD推理引擎，支持线程本地会话管理和模型预热。
"""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from cascade.types import (
    AudioChunk,
    CascadeError,
    ErrorCode,
    ModelLoadError,
    ONNXConfig,
    VADConfig,
    VADProcessingError,
    VADResult,
)

from .base import VADBackend


class ONNXVADBackend(VADBackend):
    """
    ONNX VAD后端实现
    
    提供基于ONNXRuntime的高性能VAD推理：
    - 线程本地ONNX会话管理
    - 自动模型预热机制
    - 支持CPU/GPU执行提供者
    - 线程安全的并发推理
    """

    def __init__(self, vad_config: VADConfig):
        """
        初始化ONNX VAD后端
        
        Args:
            vad_config: VAD配置对象，包含ONNX特定配置
        """
        super().__init__(vad_config)

        # 创建默认ONNX配置
        self._onnx_config = ONNXConfig()
        self._thread_local = threading.local()
        self._model_path: str | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._is_warmed_up = False
        self._warmup_lock = threading.Lock()

        # 从VAD配置中提取采样率等信息
        self._expected_sample_rate = 16000  # 默认值，实际应该从配置获取

    @property
    def onnx_config(self) -> ONNXConfig:
        """获取ONNX配置"""
        return self._onnx_config

    def _get_session(self):
        """
        获取线程本地的ONNX推理会话
        
        使用线程本地存储确保每个线程都有独立的会话实例，
        避免并发访问冲突，提高推理性能。
        
        Returns:
            ONNX Runtime推理会话
            
        Raises:
            ModelLoadError: 当会话创建失败时
        """
        if not hasattr(self._thread_local, 'session'):
            try:
                # 延迟导入避免启动时的依赖检查
                import onnxruntime as ort

                # 创建会话选项
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self._onnx_config.intra_op_num_threads
                sess_options.inter_op_num_threads = self._onnx_config.inter_op_num_threads
                sess_options.execution_mode = getattr(
                    ort.ExecutionMode,
                    self._onnx_config.execution_mode.upper(),
                    ort.ExecutionMode.ORT_SEQUENTIAL
                )
                sess_options.graph_optimization_level = getattr(
                    ort.GraphOptimizationLevel,
                    f"ORT_{self._onnx_config.graph_optimization_level.upper()}_ALL",
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )

                # 创建会话
                self._thread_local.session = ort.InferenceSession(
                    self._model_path,
                    sess_options,
                    providers=self._onnx_config.providers
                )

                # 获取输入输出名称（如果还没有获取）
                if self._input_name is None:
                    self._input_name = self._thread_local.session.get_inputs()[0].name
                if self._output_name is None:
                    self._output_name = self._thread_local.session.get_outputs()[0].name

            except Exception as e:
                raise ModelLoadError(
                    self._model_path or "unknown",
                    f"创建ONNX会话失败: {str(e)}"
                )

        return self._thread_local.session

    async def initialize(self) -> None:
        """
        异步初始化ONNX后端
        
        执行模型加载、验证和基础配置检查。
        
        Raises:
            ModelLoadError: 当模型加载失败时
            CascadeError: 当配置无效时
        """
        try:
            # 延迟导入检查ONNX Runtime可用性
            import onnxruntime as ort

            # 检查执行提供者可用性
            available_providers = ort.get_available_providers()
            for provider in self._onnx_config.providers:
                if provider not in available_providers:
                    raise CascadeError(
                        f"执行提供者不可用: {provider}",
                        ErrorCode.BACKEND_UNAVAILABLE,
                        context={"available_providers": available_providers}
                    )

            # 设置默认模型路径（实际项目中应该从配置获取）
            if self._onnx_config.model_path:
                self._model_path = self._onnx_config.model_path
            else:
                # 这里应该有默认模型或从配置中获取
                # 为了演示，我们设置一个占位符
                self._model_path = "vad_model.onnx"

            # 验证模型文件存在性
            if self._model_path and not Path(self._model_path).exists():
                raise ModelLoadError(
                    self._model_path,
                    "模型文件不存在"
                )

            # 创建一个测试会话来验证模型
            test_session = self._get_session()

            # 验证模型输入输出格式
            inputs = test_session.get_inputs()
            outputs = test_session.get_outputs()

            if len(inputs) != 1:
                raise ModelLoadError(
                    self._model_path,
                    f"期望单一输入，但模型有 {len(inputs)} 个输入"
                )

            if len(outputs) != 1:
                raise ModelLoadError(
                    self._model_path,
                    f"期望单一输出，但模型有 {len(outputs)} 个输出"
                )

            # 保存输入输出信息
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name

            self._initialized = True

        except ImportError:
            raise CascadeError(
                "ONNXRuntime未安装，请执行: pip install onnxruntime",
                ErrorCode.BACKEND_UNAVAILABLE
            )
        except Exception as e:
            if isinstance(e, (CascadeError, ModelLoadError)):
                raise
            raise ModelLoadError(
                self._model_path or "unknown",
                f"初始化失败: {str(e)}"
            )

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块并返回VAD检测结果
        
        这是核心推理方法，必须高性能且线程安全。
        
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
            # 预处理音频数据
            audio_data = self._preprocess_audio(chunk)

            # 获取线程本地会话
            session = self._get_session()

            # 准备输入数据
            ort_inputs = {self._input_name: audio_data}

            # 执行推理
            start_time = time.time()
            ort_outputs = session.run([self._output_name], ort_inputs)
            inference_time = time.time() - start_time

            # 后处理结果
            vad_result = self._postprocess_output(
                ort_outputs[0],
                chunk,
                inference_time
            )

            return vad_result

        except Exception as e:
            raise VADProcessingError(
                f"ONNX推理失败: {str(e)}",
                ErrorCode.INFERENCE_FAILED,
                context={
                    "chunk_id": chunk.sequence_number,
                    "chunk_size": chunk.chunk_size,
                    "model_path": self._model_path
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
        # 使用锁确保预热只执行一次
        with self._warmup_lock:
            if self._is_warmed_up:
                return

            try:
                # 执行预热推理
                for _ in range(self._onnx_config.warmup_iterations):
                    _ = self.process_chunk(dummy_chunk)

                self._is_warmed_up = True

            except Exception as e:
                raise VADProcessingError(
                    f"模型预热失败: {str(e)}",
                    ErrorCode.INFERENCE_FAILED
                )

    async def close(self) -> None:
        """
        异步关闭后端并释放资源
        
        清理ONNX会话和相关资源。
        """
        try:
            # 清理线程本地会话
            if hasattr(self._thread_local, 'session'):
                delattr(self._thread_local, 'session')

            self._initialized = False
            self._is_warmed_up = False

        except Exception:
            # 静默处理清理错误
            pass

    def _preprocess_audio(self, chunk: AudioChunk) -> np.ndarray:
        """
        预处理音频数据为模型输入格式
        
        Args:
            chunk: 音频数据块
            
        Returns:
            预处理后的音频数据
        """
        try:
            # 假设chunk.data是numpy数组或可转换为numpy数组
            audio_array = np.asarray(chunk.data, dtype=np.float32)

            # 确保数据是一维的
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            # 添加批次维度 [batch_size=1, sequence_length]
            audio_array = audio_array.reshape(1, -1)

            # 确保数据连续性（ONNX Runtime优化）
            audio_array = np.ascontiguousarray(audio_array)

            return audio_array

        except Exception as e:
            raise VADProcessingError(
                f"音频预处理失败: {str(e)}",
                ErrorCode.INVALID_INPUT,
                context={"chunk_shape": getattr(chunk.data, 'shape', 'unknown')}
            )

    def _postprocess_output(
        self,
        raw_output: np.ndarray,
        chunk: AudioChunk,
        inference_time: float
    ) -> VADResult:
        """
        后处理模型输出为VAD结果
        
        Args:
            raw_output: 模型原始输出
            chunk: 输入音频块
            inference_time: 推理耗时
            
        Returns:
            处理后的VAD结果
        """
        try:
            # 假设模型输出是概率值 [batch_size, num_classes]
            # 对于二分类VAD：[batch_size, 2] 或 [batch_size, 1]

            if raw_output.ndim == 2 and raw_output.shape[1] == 2:
                # 二分类输出 [batch_size, 2]
                speech_prob = float(raw_output[0, 1])
            elif raw_output.ndim == 2 and raw_output.shape[1] == 1:
                # 单一概率输出 [batch_size, 1]
                speech_prob = float(raw_output[0, 0])
            else:
                # 假设是标量概率
                speech_prob = float(raw_output.item())

            # 应用阈值判断
            is_speech = speech_prob >= self._config.threshold

            # 计算置信度（这里简化处理）
            confidence = speech_prob if is_speech else (1.0 - speech_prob)

            return VADResult(
                is_speech=is_speech,
                probability=speech_prob,
                start_ms=chunk.timestamp_ms,
                end_ms=chunk.get_end_timestamp_ms(),
                chunk_id=chunk.sequence_number,
                confidence=confidence,
                metadata={
                    "inference_time_ms": inference_time * 1000,
                    "model_path": self._model_path,
                    "backend": "onnx"
                }
            )

        except Exception as e:
            raise VADProcessingError(
                f"输出后处理失败: {str(e)}",
                ErrorCode.RESULT_VALIDATION_FAILED,
                context={
                    "output_shape": raw_output.shape,
                    "output_dtype": str(raw_output.dtype)
                }
            )

    def get_backend_info(self) -> dict[str, Any]:
        """
        获取ONNX后端详细信息
        
        Returns:
            包含后端信息的字典
        """
        info = super().get_backend_info()
        info.update({
            "onnx_config": self._onnx_config.__dict__,
            "model_path": self._model_path,
            "input_name": self._input_name,
            "output_name": self._output_name,
            "is_warmed_up": self._is_warmed_up,
            "expected_sample_rate": self._expected_sample_rate
        })

        try:
            import onnxruntime as ort
            info["onnxruntime_version"] = ort.__version__
            info["available_providers"] = ort.get_available_providers()
        except ImportError:
            info["onnxruntime_version"] = "not_installed"
            info["available_providers"] = []

        return info
