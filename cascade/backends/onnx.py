"""
基于ONNX Runtime的VAD后端实现模块

本模块实现了基于ONNX Runtime的高性能VAD后端，
利用线程本地存储(threading.local)实现"一个线程一个VAD实例"的设计模式，
确保在多线程环境下的高性能无锁并行推理。
"""

import os
import threading
from typing import List, Dict, Any, Optional

import numpy as np
import onnxruntime as ort

from cascade.backends.base import VADBackend
from cascade.types.audio import AudioChunk
from cascade.types.config import ONNXConfig
from cascade.types.errors import ModelLoadError
from cascade.types.vad import VADResult


class ONNXVADBackend(VADBackend):
    """
    基于ONNX Runtime的VAD后端实现。
    
    本类利用线程本地存储(threading.local)来实现"一个线程一个VAD实例"的设计模式。
    它正确地处理了有状态的RNN模型（如Silero VAD），在每个线程中独立管理模型的
    循环状态 (state)，实现了无锁的高性能并行推理。
    """

    def __init__(self, config: ONNXConfig):
        """
        初始化ONNX VAD后端。
        
        Args:
            config: ONNX后端的配置对象。
        """
        super().__init__(config)
        self._thread_local = threading.local()
        self._main_thread_initialized = False
        if not isinstance(config, ONNXConfig):
            raise TypeError(f"配置必须是ONNXConfig类型，而不是{type(config)}")
        self.config: ONNXConfig = config

    def initialize(self) -> None:
        """
        在主线程中执行，用于验证配置的有效性。
        """
        if not self.config.model_path:
            raise ModelLoadError("", "模型路径未设置")
        
        if not os.path.exists(self.config.model_path):
            raise ModelLoadError(self.config.model_path, "模型文件不存在")
            
        if self.config.threshold < 0 or self.config.threshold > 1:
            raise ValueError(f"阈值必须在0到1之间，当前值为{self.config.threshold}")
            
        self._is_initialized = True
        self._main_thread_initialized = True

    def _get_or_create_session_and_state(self) -> ort.InferenceSession:
        """
        获取或创建当前线程的ONNX推理会话及相关状态。
        
        这是实现"线程池-实例池"模式并管理RNN状态的核心。
        如果当前线程没有会话，则创建一个新的会话，并初始化
        RNN的状态张量(state)和采样率张量(sr)。
        
        Returns:
            当前线程的ONNX推理会话。
            
        Raises:
            ModelLoadError: 当模型初始化失败时抛出。
        """
        if not self._main_thread_initialized:
            raise ModelLoadError("", "必须先在主线程中调用initialize()方法")
            
        if not hasattr(self._thread_local, 'session'):
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self.config.intra_op_num_threads
                sess_options.inter_op_num_threads = self.config.inter_op_num_threads
                if hasattr(self.config, 'graph_optimization_level'):
                    sess_options.graph_optimization_level = self.config.graph_optimization_level
                
                session = ort.InferenceSession(
                    self.config.model_path,
                    providers=self.config.providers,
                    sess_options=sess_options,
                )
                self._thread_local.session = session

                # 初始化RNN状态
                self._thread_local.state = np.zeros((2, 1, 128), dtype=np.float32)
                self._thread_local.sr = np.array([self.config.sample_rate], dtype=np.int64)

            except Exception as e:
                raise ModelLoadError(self.config.model_path, f"为线程{threading.current_thread().name}创建ONNX会话失败: {str(e)}")
                
        return self._thread_local.session

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块，执行VAD推理。
        """
        try:
            session = self._get_or_create_session_and_state()
            
            input_data = chunk.data.astype(np.float32)
            # The model expects a 2D input: (batch_size, num_samples)
            if len(input_data.shape) == 1:
                input_data = np.expand_dims(input_data, axis=0)  # Shape: (N,) -> (1, N)

            inputs = {
                'input': input_data,
                'sr': self._thread_local.sr,
                'state': self._thread_local.state,
            }
            
            output, next_state = session.run(None, inputs)
            
            # 更新状态
            self._thread_local.state = next_state
            
            probability = float(output[0])
            
            return VADResult(
                is_speech=(probability > self.config.threshold),
                probability=probability,
                start_ms=chunk.timestamp_ms,
                end_ms=chunk.timestamp_ms + chunk.get_duration_ms(),
                chunk_id=chunk.sequence_number,
                confidence=probability,
            )
            
        except Exception as e:
            raise ModelLoadError(self.config.model_path, f"处理音频块失败: {str(e)}")

    def warmup(self) -> None:
        """
        预热模型，消除首次推理的延迟。
        """
        try:
            session = self._get_or_create_session_and_state()
            
            chunk_size = int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)
            dummy_input = np.zeros((1, chunk_size), dtype=np.float32)
            
            inputs = {
                'input': dummy_input,
                'sr': self._thread_local.sr,
                'state': self._thread_local.state,
            }
            
            # 在预热时也更新状态，确保状态有效
            _, next_state = session.run(None, inputs)
            self._thread_local.state = next_state
            
        except Exception as e:
            raise ModelLoadError(self.config.model_path, f"模型预热失败: {str(e)}")

    def close(self) -> None:
        """关闭后端，释放资源。"""
        pass