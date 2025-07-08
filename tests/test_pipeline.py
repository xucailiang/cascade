"""
处理管道测试
"""

import asyncio
from typing import Any

import numpy as np
import pytest

from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
)
from cascade.processor.pipeline import (
    AudioPipeline,
    ParallelPipeline,
    PipelineConfig,
    PipelineResult,
    ProcessorNode,
    SequentialPipeline,
)
from cascade.types.audio import AudioConfig, AudioFormat


class MockProcessor(AudioProcessor):
    """模拟处理器，用于测试"""

    def __init__(self, return_value: Any = None, raise_error: bool = False):
        super().__init__()
        self.return_value = return_value
        self.raise_error = raise_error
        self.processed_chunks = []

    def process_chunk(self, chunk: AudioChunk) -> Any:
        """处理单个音频块"""
        self.processed_chunks.append(chunk)

        if self.raise_error:
            raise ValueError("模拟处理错误")

        return self.return_value


# 测试专用的管道类，实现process_chunk方法
class TestAudioPipeline(AudioPipeline):
    """测试用的音频处理管道"""
    def process_chunk(self, chunk: AudioChunk) -> Any:
        """实现抽象方法，但实际上不会被调用"""
        return None


class TestSequentialPipeline(SequentialPipeline):
    """测试用的串行处理管道"""
    def process_chunk(self, chunk: AudioChunk) -> Any:
        """实现抽象方法，但实际上不会被调用"""
        return None


class TestParallelPipeline(ParallelPipeline):
    """测试用的并行处理管道"""
    def process_chunk(self, chunk: AudioChunk) -> Any:
        """实现抽象方法，但实际上不会被调用"""
        return None


@pytest.fixture
def sample_audio():
    """生成测试用的音频数据"""
    # 创建一个简单的合成音频
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 创建一个简单的正弦波
    t = np.linspace(0, duration_sec, num_samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

    return audio, sample_rate


@pytest.fixture
def audio_config(sample_audio):
    """创建音频配置"""
    _, sample_rate = sample_audio
    return AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )


@pytest.mark.asyncio
async def test_sequential_pipeline_basic(sample_audio, audio_config):
    """测试串行处理管道的基本功能"""
    audio_data, _ = sample_audio

    # 创建两个模拟处理器
    processor1 = MockProcessor(return_value="result1")
    processor2 = MockProcessor(return_value="result2")

    # 创建串行处理管道
    pipeline = TestSequentialPipeline(
        name="test_sequential",
        processors=[
            ProcessorNode(processor=processor1, name="proc1"),
            ProcessorNode(processor=processor2, name="proc2")
        ]
    )

    # 处理音频
    results = await pipeline.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) == 1
    assert isinstance(results[0], PipelineResult)
    assert results[0].pipeline_name == "test_sequential"
    assert "proc1" in results[0].processor_results
    assert "proc2" in results[0].processor_results

    # 验证处理顺序（串行处理中，第二个处理器应该处理与第一个处理器相同数量的块）
    assert len(processor1.processed_chunks) > 0
    assert len(processor1.processed_chunks) == len(processor2.processed_chunks)


@pytest.mark.asyncio
async def test_parallel_pipeline_basic(sample_audio, audio_config):
    """测试并行处理管道的基本功能"""
    audio_data, _ = sample_audio

    # 创建两个模拟处理器
    processor1 = MockProcessor(return_value="result1")
    processor2 = MockProcessor(return_value="result2")

    # 创建并行处理管道
    pipeline = TestParallelPipeline(
        name="test_parallel",
        processors=[
            ProcessorNode(processor=processor1, name="proc1"),
            ProcessorNode(processor=processor2, name="proc2")
        ]
    )

    # 处理音频
    results = await pipeline.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) == 1
    assert isinstance(results[0], PipelineResult)
    assert results[0].pipeline_name == "test_parallel"
    assert "proc1" in results[0].processor_results
    assert "proc2" in results[0].processor_results

    # 验证处理（并行处理中，两个处理器应该处理相同数量的块）
    assert len(processor1.processed_chunks) > 0
    assert len(processor1.processed_chunks) == len(processor2.processed_chunks)


@pytest.mark.asyncio
async def test_processor_node_management(sample_audio, audio_config):
    """测试处理器节点管理"""
    audio_data, _ = sample_audio

    # 创建处理管道
    pipeline = TestAudioPipeline(PipelineConfig(name="test_management"))

    # 添加处理器
    processor1 = MockProcessor(return_value="result1")
    processor2 = MockProcessor(return_value="result2")

    pipeline.add_processor(processor1, "proc1")
    pipeline.add_processor(processor2, "proc2")

    # 验证处理器添加成功
    assert len(pipeline.processors) == 2
    assert pipeline.get_processor("proc1") is not None
    assert pipeline.get_processor("proc2") is not None

    # 测试禁用处理器
    pipeline.enable_processor("proc2", False)
    results = await pipeline.process_audio(audio_data, audio_config)

    # 只有proc1应该被处理
    assert "proc1" in results[0].processor_results
    assert "proc2" not in results[0].processor_results

    # 测试移除处理器
    assert pipeline.remove_processor("proc1") is True
    assert len(pipeline.processors) == 1
    assert pipeline.get_processor("proc1") is None


@pytest.mark.asyncio
async def test_input_output_transform(sample_audio, audio_config):
    """测试输入和输出转换"""
    audio_data, _ = sample_audio

    # 创建模拟处理器
    processor = MockProcessor(return_value="original_result")

    # 定义输入转换函数
    def input_transform(audio, config):
        # 将音频数据乘以2
        return audio * 2, config

    # 定义输出转换函数
    def output_transform(results):
        # 修改结果
        for result in results:
            result.result_data = "transformed_" + str(result.result_data)
        return results

    # 创建处理管道
    pipeline = TestSequentialPipeline(
        name="test_transform",
        processors=[
            ProcessorNode(
                processor=processor,
                name="proc",
                input_transform=input_transform,
                output_transform=output_transform
            )
        ]
    )

    # 处理音频
    results = await pipeline.process_audio(audio_data, audio_config)

    # 验证结果
    assert "proc" in results[0].processor_results

    # 验证输出转换生效
    for result in results[0].processor_results["proc"]:
        assert result.result_data.startswith("transformed_")

    # 验证输入转换生效（通过检查处理器处理的块数据是否被放大）
    for chunk in processor.processed_chunks:
        # 由于输入转换将音频数据乘以2，处理的块的最大值应该大于原始音频的最大值
        assert np.max(np.abs(chunk.data)) > np.max(np.abs(audio_data))


@pytest.mark.asyncio
async def test_result_aggregator(sample_audio, audio_config):
    """测试结果聚合"""
    audio_data, _ = sample_audio

    # 创建两个模拟处理器
    processor1 = MockProcessor(return_value={"count": 1})
    processor2 = MockProcessor(return_value={"count": 2})

    # 定义结果聚合函数
    def result_aggregator(processor_results):
        total_count = 0
        for processor_name, results in processor_results.items():
            for result in results:
                if isinstance(result.result_data, dict) and "count" in result.result_data:
                    total_count += result.result_data["count"]
        return {"total_count": total_count}

    # 创建处理管道
    pipeline = TestSequentialPipeline(
        name="test_aggregator",
        processors=[
            ProcessorNode(processor=processor1, name="proc1"),
            ProcessorNode(processor=processor2, name="proc2")
        ],
        result_aggregator=result_aggregator
    )

    # 处理音频
    results = await pipeline.process_audio(audio_data, audio_config)

    # 验证结果
    assert results[0].aggregated_result is not None
    assert "total_count" in results[0].aggregated_result
    assert results[0].aggregated_result["total_count"] > 0


@pytest.mark.asyncio
async def test_error_handling(sample_audio, audio_config):
    """测试错误处理"""
    audio_data, _ = sample_audio

    # 创建两个处理器，第一个会抛出错误
    processor1 = MockProcessor(raise_error=True)
    processor2 = MockProcessor(return_value="result2")

    # 创建不继续执行的处理管道
    pipeline1 = TestSequentialPipeline(
        name="test_error_stop",
        continue_on_error=False,
        processors=[
            ProcessorNode(processor=processor1, name="proc1"),
            ProcessorNode(processor=processor2, name="proc2")
        ]
    )

    # 处理音频
    results1 = await pipeline1.process_audio(audio_data, audio_config)

    # 验证结果（第二个处理器不应该被执行）
    assert "proc1" in results1[0].processor_results
    assert "proc2" not in results1[0].processor_results

    # 创建继续执行的处理管道
    pipeline2 = TestSequentialPipeline(
        name="test_error_continue",
        continue_on_error=True,
        processors=[
            ProcessorNode(processor=processor1, name="proc1"),
            ProcessorNode(processor=processor2, name="proc2")
        ]
    )

    # 处理音频
    results2 = await pipeline2.process_audio(audio_data, audio_config)

    # 验证结果（第二个处理器应该被执行）
    assert "proc1" in results2[0].processor_results
    assert "proc2" in results2[0].processor_results


@pytest.mark.asyncio
async def test_empty_audio(audio_config):
    """测试处理空音频"""
    # 创建空音频
    audio_data = np.array([], dtype=np.float32)

    # 创建处理管道
    processor = MockProcessor(return_value="result")
    pipeline = TestSequentialPipeline(
        name="test_empty",
        processors=[
            ProcessorNode(processor=processor, name="proc")
        ]
    )

    # 处理音频
    results = await pipeline.process_audio(audio_data, audio_config)

    # 验证结果（应该没有结果）
    assert len(results) == 1
    assert not processor.processed_chunks


@pytest.mark.asyncio
async def test_timeout(sample_audio, audio_config):
    """测试处理超时"""
    audio_data, _ = sample_audio

    # 创建一个会睡眠的处理器
    class SlowProcessor(AudioProcessor):
        def process_chunk(self, chunk):
            # 模拟耗时处理
            import time
            time.sleep(0.5)
            return "slow_result"

    # 创建带超时的处理管道
    pipeline = TestSequentialPipeline(
        name="test_timeout",
        timeout_seconds=0.1,  # 设置一个很短的超时
        processors=[
            ProcessorNode(processor=SlowProcessor(), name="slow_proc")
        ]
    )

    # 处理音频（应该会超时）
    with pytest.raises(asyncio.TimeoutError):
        await pipeline.process_audio(audio_data, audio_config)


if __name__ == "__main__":
    asyncio.run(test_sequential_pipeline_basic(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_parallel_pipeline_basic(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_processor_node_management(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_input_output_transform(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_result_aggregator(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_error_handling(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_empty_audio(audio_config(sample_audio())))
    print("所有测试通过！")
