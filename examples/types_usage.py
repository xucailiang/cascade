"""
类型系统使用示例

本示例展示了如何使用Cascade的类型系统，包括：
- 音频配置和数据块
- VAD配置和结果
- 错误处理
- 状态管理
"""

import os
import sys

import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cascade.types import (
    AudioChunk,
    # 音频相关类型
    AudioConfig,
    AudioFormat,
    AudioFormatError,
    ErrorInfo,
    # 后端配置类型
    ONNXConfig,
    OptimizationLevel,
    PerformanceMetrics,
    # 错误类型
    PreVADError,
    ProcessingMode,
    # 通用类型
    Status,
    VADBackend,
    VADConfig,
    VADResult,
    VLLMConfig,
)


def audio_config_example():
    """音频配置示例"""
    print("\n=== 音频配置示例 ===")

    # 创建默认配置
    default_config = AudioConfig()
    print(f"默认配置: {default_config.json(indent=2)}")

    # 创建自定义配置
    custom_config = AudioConfig(
        sample_rate=44100,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32",
        bit_depth=24
    )
    print(f"自定义配置: {custom_config.json(indent=2)}")

    # 使用配置方法
    frame_size = custom_config.get_frame_size(500)  # 500ms的帧大小
    print(f"500ms在44.1kHz下的帧大小: {frame_size}样本")

    bytes_per_second = custom_config.get_bytes_per_second()
    print(f"每秒字节数: {bytes_per_second} bytes/s")

    # 验证规则示例
    try:
        AudioConfig(sample_rate=10000)  # 不支持的采样率
    except Exception as e:
        print(f"验证错误: {str(e)}")


def audio_chunk_example():
    """音频数据块示例"""
    print("\n=== 音频数据块示例 ===")

    # 创建一个简单的音频块
    sample_rate = 16000
    duration_ms = 500
    samples = int(sample_rate * duration_ms / 1000)

    # 生成一个正弦波作为示例数据
    t = np.linspace(0, duration_ms/1000, samples, False)
    data = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    chunk = AudioChunk(
        data=data,
        sequence_number=1,
        start_frame=0,
        chunk_size=samples,
        overlap_size=160,  # 10ms at 16kHz
        timestamp_ms=0.0,
        sample_rate=sample_rate,
        is_last=False
    )

    # 使用数据块方法
    total_size = chunk.get_total_size()
    print(f"总大小: {total_size}样本")

    duration = chunk.get_duration_ms()
    print(f"块时长: {duration}ms")

    end_timestamp = chunk.get_end_timestamp_ms()
    print(f"结束时间戳: {end_timestamp}ms")


def vad_config_example():
    """VAD配置示例"""
    print("\n=== VAD配置示例 ===")

    # 创建默认配置
    default_config = VADConfig()
    print(f"默认配置: {default_config.json(indent=2)}")

    # 创建自定义配置
    custom_config = VADConfig(
        backend=VADBackend.ONNX,
        workers=8,
        threshold=0.7,
        chunk_duration_ms=1000,
        overlap_ms=32,
        buffer_capacity_seconds=10,
        processing_mode=ProcessingMode.BATCH,
        optimization_level=OptimizationLevel.BASIC,
        min_speech_duration_ms=200,
        max_silence_duration_ms=1000,
        energy_threshold=0.1,
        smoothing_window_ms=100
    )
    print(f"自定义配置: {custom_config.json(indent=2)}")

    # 使用配置方法
    chunk_samples = custom_config.get_chunk_samples(16000)
    print(f"块样本数 (16kHz): {chunk_samples}")

    overlap_samples = custom_config.get_overlap_samples(16000)
    print(f"重叠样本数 (16kHz): {overlap_samples}")


def vad_result_example():
    """VAD结果示例"""
    print("\n=== VAD结果示例 ===")

    # 创建VAD结果
    result = VADResult(
        is_speech=True,
        probability=0.85,
        start_ms=1000.0,
        end_ms=1500.0,
        chunk_id=2,
        confidence=0.9,
        energy_level=0.7,
        snr_db=15.0,
        speech_type="male"
    )
    print(f"VAD结果: {result.json(indent=2)}")

    # 使用结果方法
    duration = result.get_duration_ms()
    print(f"语音段时长: {duration}ms")

    speech_ratio = result.get_speech_ratio()
    print(f"语音比例: {speech_ratio}")

    is_high_confidence = result.is_high_confidence()
    print(f"是否高置信度: {is_high_confidence}")


def status_example():
    """状态示例"""
    print("\n=== 状态示例 ===")

    # 创建成功状态
    ok_status = Status.ok("操作成功", {"operation": "audio_processing"})
    print(f"成功状态: {ok_status.json(indent=2)}")

    # 创建错误状态
    error_status = Status.error(404, "资源不存在", {"resource_id": "123"})
    print(f"错误状态: {error_status.json(indent=2)}")

    # 使用状态方法
    print(f"状态是否正常: {ok_status.is_ok()}")
    print(f"状态是否错误: {error_status.is_error()}")


def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    try:
        # 模拟音频格式错误
        raise AudioFormatError(
            "不支持的音频格式: MP3",
            {"format": "mp3", "sample_rate": 44100}
        )
    except PreVADError as e:
        # 处理错误
        print(f"捕获到错误: {e.message}")
        print(f"错误码: {e.error_code}")
        print(f"严重程度: {e.severity}")
        print(f"上下文: {e.context}")

        # 转换为错误信息对象
        error_info = ErrorInfo.from_exception(e)
        print(f"错误信息: {error_info.json(indent=2)}")


def backend_config_example():
    """后端配置示例"""
    print("\n=== 后端配置示例 ===")

    # 创建ONNX配置
    onnx_config = ONNXConfig(
        model_path="/path/to/model.onnx",
        device="cuda",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        intra_op_num_threads=4,
        inter_op_num_threads=2,
        execution_mode="parallel",
        graph_optimization_level="basic"
    )
    print(f"ONNX配置: {onnx_config.json(indent=2)}")

    # 创建VLLM配置
    vllm_config = VLLMConfig(
        model_path="/path/to/model",
        device="cuda",
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        swap_space=8,
        dtype="float16"
    )
    print(f"VLLM配置: {vllm_config.json(indent=2)}")


def performance_metrics_example():
    """性能指标示例"""
    print("\n=== 性能指标示例 ===")

    # 创建性能指标
    metrics = PerformanceMetrics(
        avg_latency_ms=10.5,
        p50_latency_ms=8.2,
        p95_latency_ms=15.3,
        p99_latency_ms=20.1,
        max_latency_ms=25.0,
        throughput_qps=100.0,
        throughput_mbps=5.0,
        error_rate=0.01,
        success_count=990,
        error_count=10,
        memory_usage_mb=256.0,
        cpu_usage_percent=45.0,
        active_threads=4,
        queue_depth=10,
        buffer_utilization=0.5,
        zero_copy_rate=0.8,
        cache_hit_rate=0.9,
        collection_duration_seconds=60.0
    )
    print(f"性能指标: {metrics.json(indent=2)}")

    # 使用指标方法
    total_ops = metrics.get_total_operations()
    print(f"总操作数: {total_ops}")

    success_rate = metrics.get_success_rate()
    print(f"成功率: {success_rate}")

    is_healthy = metrics.is_healthy()
    print(f"性能是否健康: {is_healthy}")


def main():
    """主函数"""
    print("Cascade类型系统使用示例")

    audio_config_example()
    audio_chunk_example()
    vad_config_example()
    vad_result_example()
    status_example()
    error_handling_example()
    backend_config_example()
    performance_metrics_example()


if __name__ == "__main__":
    main()
