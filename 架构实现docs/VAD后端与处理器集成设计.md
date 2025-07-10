# VAD ONNX后端与处理器集成设计

## 1. 目标与原则

本设计方案旨在填补当前架构中缺失的核心VAD功能，通过引入一个可插拔的、基于ONNX的高性能VAD后端，并将其无缝集成到现有的音频处理管道中。

**设计原则**:

-   **遵循现有架构**: 继承并利用现有的 `types` 模块、`processor` 管道模式以及 `buffer` 和 `formats` 模块的优秀设计。
-   **依赖倒置**: `processor` 模块应依赖于 `VADBackend` 抽象基类，而不是具体的ONNX实现，以保持系统的可扩展性。
-   **高性能并发**: 保留并优化设计文档中提出的“线程池 + 线程本地模型实例”的核心思想，确保低延迟和高吞吐。
-   **配置驱动**: 所有后端和处理器的行为都应通过类型安全、验证完善的 `pydantic` 配置模型进行控制。
-   **模块化与高内聚**: `backends` 模块应独立、高内聚，只负责模型推理；`processor` 负责业务编排；`ResultMerger` 负责后处理。

## 2. 模块依赖关系

新的依赖关系将严格遵循单向数据流，`processor` 作为协调者，依赖于 `backends` 和其他底层模块。

```mermaid
graph TD
    subgraph "核心类型 (cascade.types)"
        A[types.audio]
        B[types.vad]
        C[types.config]
        D[types.generic]
        E[types.errors]
    end

    subgraph "底层工具 (cascade._internal)"
        F[thread_pool.py]
        G[performance.py]
        H[atomic.py]
    end

    subgraph "数据输入 (cascade.formats & cascade.buffer)"
        I[formats.base]
        J[buffer.base]
    end
    
    subgraph "VAD后端 (cascade.backends)"
        K[base.py<br/>VADBackend (ABC)]
        L[onnx.py<br/>ONNXVADBackend]
    end

    subgraph "核心处理器 (cascade.processor)"
        M[pipeline.py<br/>AudioPipeline]
        N[base.py<br/>AudioProcessor]
        O[vad_processor.py<br/>VADProcessor]
        P[merger.py<br/>ResultMerger]
    end
    
    subgraph "顶层API (cascade)"
        Q[__init__.py]
    end

    A & B & C & D & E --- I
    A & B & C & D & E --- J
    A & B & C & D & E --- K
    A & B & C & D & E --- L
    A & B & C & D & E --- O
    A & B & C & D & E --- P

    F & G & H --- L
    F & G & H --- O
    
    I ---> O
    J ---> O
    K ---> L
        
    K ---> O
    L -.-> O
    P ---> O
    F ---> O
    
    O ---> M
    
    O ---> Q
    B ---> Q
    C ---> Q
    L ---> Q
```
**关键依赖**: `VADProcessor`(`O`) 依赖于 `VADBackend` 抽象(`K`)，而不是具体实现 `ONNXVADBackend`(`L`)。

## 3. 核心组件设计

### 3.1 VAD后端抽象 (`backends/base.py`)

定义所有VAD后端必须遵守的接口。

```python
# cascade/backends/base.py
from abc import ABC, abstractmethod
from typing import List

from cascade.types.audio import AudioChunk
from cascade.types.config import BackendConfig
from cascade.types.vad import VADResult

class VADBackend(ABC):
    """VAD后端模块的抽象基类"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
        
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
```

### 3.2 ONNX后端实现 (`backends/onnx.py`)

基于ONNX Runtime的高性能后端实现。

```python
# cascade/backends/onnx.py
import os
import threading
from typing import List

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
    
    本类利用线程本地存储(threading.local)来实现“一个线程一个VAD实例”的设计模式。
    VADProcessor会创建本类的一个实例（作为模板），并传递给VADThreadPool。
    当线程池中的每个线程首次执行任务时，会通过_get_session()为自己创建一个独立的
    ONNX InferenceSession，从而实现无锁的高性能并行推理。
    """

    def __init__(self, config: ONNXConfig):
        super().__init__(config)
        # 线程本地存储，用于为每个工作线程维护一个独立的推理会话。
        self._thread_local = threading.local()
        # 主线程中的初始化检查标记，防止在没有有效配置的情况下启动。
        self._main_thread_initialized = False

    def initialize(self) -> None:
        """
        在主线程中执行，用于验证配置的有效性。
        实际的模型加载和会话创建将延迟到每个工作线程首次需要时进行。
        """
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise ModelLoadError(self.config.model_path, "模型文件不存在或路径未设置")
        self._is_initialized = True

    def _get_session(self) -> ort.InferenceSession:
        """
        获取或创建当前线程的ONNX推理会话。
        这是实现“线程池-实例池”模式(1:1绑定)的核心所在。
        - 检查当前线程的本地存储中是否存在'session'属性。
        - 如果不存在，说明这是该线程第一次执行此操作。
        - 则为该线程创建一个新的InferenceSession实例，并存储到其本地存储中。
        - 如果存在，直接返回已创建的实例。
        """
        if not hasattr(self._thread_local, 'session'):
            # print(f"为线程创建新的ONNX会话: {threading.current_thread().name}")
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.intra_op_num_threads
            sess_options.inter_op_num_threads = self.config.inter_op_num_threads

            self._thread_local.session = ort.InferenceSession(
                self.config.model_path,
                providers=self.config.providers,
                sess_options=sess_options,
            )
        return self._thread_local.session

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        session = self._get_session()
        input_data = chunk.data.astype(np.float32)
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
        input_name = session.get_inputs()[0].name
        inputs = {input_name: input_data}
        outputs = session.run(None, inputs)
        probability = float(outputs[0][0])
        
        result = VADResult(
            is_speech=(probability > self.config.threshold),
            probability=probability,
            start_ms=chunk.timestamp_ms,
            end_ms=chunk.timestamp_ms + chunk.get_duration_ms(),
            chunk_id=chunk.sequence_number,
            confidence=probability,
        )
        return result

    def warmup(self) -> None:
        session = self._get_session()
        input_cfg = session.get_inputs()[0]
        input_shape = [1, self.config.chunk_duration_ms * 16] # 假设16kHz采样率
        dummy_input = np.zeros(input_shape, dtype=np.float32)
        session.run(None, {input_cfg.name: dummy_input})

    def close(self) -> None:
        pass
```

## 4. 处理器与线程池集成

### 4.1 线程池 (`_internal/thread_pool.py`)

`VADThreadPool` 负责管理一组工作线程，并将音频处理任务分发给它们。它的核心职责是与 `VADBackend` 协同工作，确保每个线程都有一个独立的VAD模型实例。

**实现细节**:
1.  `__init__`: 接收一个 `VADBackend` 的“模板”实例。它自身不持有模型，只持有对这个模板的引用。
2.  `start`: 启动时，它会向线程池提交N个（N=工作线程数）`_warmup_thread` 任务。
3.  `_warmup_thread`: 这个函数是关键。当每个工作线程执行它时，会调用 `backend_template.warmup()`。由于 `ONNXVADBackend` 中的 `warmup` 方法会调用 `_get_session`，这就会触发在该线程本地存储中**创建并缓存**一个 `InferenceSession` 实例。这个过程会为池中的每个线程都预先创建一个模型实例。
4.  `process_chunk`: 当外部调用此方法处理音频块时，它只是简单地将任务（`_process_chunk_sync` 和 `chunk` 数据）再次提交给线程池。
5.  `_process_chunk_sync`: 在工作线程中执行时，它调用 `backend_template.process_chunk(chunk)`。因为此时当前线程已经有了自己的 `InferenceSession`，所以它会使用这个专属实例进行无锁推理。

```python
# cascade/_internal/thread_pool.py (设计)
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from cascade.backends.base import VADBackend
from cascade.types.audio import AudioChunk
from cascade.types.vad import VADConfig, VADResult

class VADThreadPool:
    def __init__(self, config: VADConfig, backend_template: VADBackend):
        self.config = config
        self._backend_template = backend_template
        self.executor = ThreadPoolExecutor(
            max_workers=config.workers,
            thread_name_prefix="vad-worker-"
        )
        self._loop = asyncio.get_event_loop()

    async def start(self):
        warmup_futures = [
            self._loop.run_in_executor(self.executor, self._warmup_thread)
            for _ in range(self.config.workers)
        ]
        await asyncio.gather(*warmup_futures)

    def _warmup_thread(self):
        self._backend_template.warmup()

    async def process_chunk(self, chunk: AudioChunk) -> VADResult:
        return await self._loop.run_in_executor(
            self.executor,
            self._process_chunk_sync,
            chunk
        )

    def _process_chunk_sync(self, chunk: AudioChunk) -> VADResult:
        return self._backend_template.process_chunk(chunk)

    async def close(self):
        self.executor.shutdown(wait=True)
        self._backend_template.close()
```

(后续部分与之前相同，省略)