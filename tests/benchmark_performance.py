#!/usr/bin/env python3
"""
Cascade 1:1:1架构性能基准测试

测试重构后的性能表现，包括：
- 处理延迟
- 吞吐量
- 内存使用
- CPU使用率
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import psutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cascade
from cascade.stream.processor import StreamProcessor


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    duration_seconds: float
    total_frames: int
    frames_per_second: float
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.process = psutil.Process()

    def generate_test_audio(self, duration_seconds: float, sample_rate: int = 16000) -> bytes:
        """生成测试音频数据"""
        samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, samples, False)

        # 生成混合频率的音频信号
        audio_data = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # 440Hz
            0.2 * np.sin(2 * np.pi * 880 * t) +  # 880Hz
            0.1 * np.random.normal(0, 0.1, samples)  # 噪声
        )

        # 转换为int16格式
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        return audio_bytes

    def measure_system_resources(self) -> dict[str, float]:
        """测量系统资源使用"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()

        return {
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': cpu_percent
        }

    async def benchmark_single_instance(self, duration: float = 5.0) -> BenchmarkResult:
        """测试单实例处理性能"""
        print(f"🔄 开始单实例性能测试 ({duration}秒音频)")

        # 生成测试音频
        audio_data = self.generate_test_audio(duration)

        # 测量开始时的资源
        start_resources = self.measure_system_resources()

        # 开始测试
        start_time = time.time()
        latencies = []
        total_frames = 0
        success_count = 0

        try:
            async for result in cascade.process_audio_file(audio_data):
                frame_time = time.time()
                latency_ms = (frame_time - start_time) * 1000
                latencies.append(latency_ms)
                total_frames += 1
                success_count += 1

                # 限制测试时间
                if time.time() - start_time > duration + 2:
                    break

        except Exception as e:
            print(f"❌ 测试过程中出错: {e}")

        end_time = time.time()
        test_duration = end_time - start_time

        # 测量结束时的资源
        end_resources = self.measure_system_resources()

        # 计算统计数据
        fps = total_frames / test_duration if test_duration > 0 else 0
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        min_latency = np.min(latencies) if latencies else 0
        success_rate = success_count / total_frames if total_frames > 0 else 0

        result = BenchmarkResult(
            test_name="单实例处理",
            duration_seconds=test_duration,
            total_frames=total_frames,
            frames_per_second=fps,
            avg_latency_ms=float(avg_latency),
            max_latency_ms=float(max_latency),
            min_latency_ms=float(min_latency),
            memory_usage_mb=end_resources['memory_mb'],
            cpu_usage_percent=end_resources['cpu_percent'],
            success_rate=success_rate
        )

        self.results.append(result)
        print(f"✅ 单实例测试完成: {total_frames}帧, {fps:.1f}fps")
        return result

    async def benchmark_stream_processor(self, duration: float = 5.0) -> BenchmarkResult:
        """测试StreamProcessor性能"""
        print(f"🔄 开始StreamProcessor性能测试 ({duration}秒音频)")

        # 生成测试音频
        audio_data = self.generate_test_audio(duration)

        # 创建配置和处理器
        config = cascade.create_default_config()
        processor = StreamProcessor(config)

        # 测量开始时的资源
        start_resources = self.measure_system_resources()

        await processor.start()

        # 开始测试
        start_time = time.time()
        latencies = []
        total_frames = 0
        success_count = 0

        try:
            # 分块处理
            chunk_size = 1024  # 1KB chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]

                chunk_start = time.time()
                results = await processor.process_chunk(chunk)
                chunk_end = time.time()

                latency_ms = (chunk_end - chunk_start) * 1000
                latencies.append(latency_ms)
                total_frames += len(results)
                success_count += len(results)

                # 限制测试时间
                if time.time() - start_time > duration + 2:
                    break

        except Exception as e:
            print(f"❌ 测试过程中出错: {e}")

        finally:
            await processor.stop()

        end_time = time.time()
        test_duration = end_time - start_time

        # 测量结束时的资源
        end_resources = self.measure_system_resources()

        # 计算统计数据
        fps = total_frames / test_duration if test_duration > 0 else 0
        avg_latency = np.mean(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        min_latency = np.min(latencies) if latencies else 0
        success_rate = success_count / max(total_frames, 1)

        result = BenchmarkResult(
            test_name="StreamProcessor",
            duration_seconds=test_duration,
            total_frames=total_frames,
            frames_per_second=fps,
            avg_latency_ms=float(avg_latency),
            max_latency_ms=float(max_latency),
            min_latency_ms=float(min_latency),
            memory_usage_mb=end_resources['memory_mb'],
            cpu_usage_percent=end_resources['cpu_percent'],
            success_rate=success_rate
        )

        self.results.append(result)
        print(f"✅ StreamProcessor测试完成: {total_frames}帧, {fps:.1f}fps")
        return result

    async def benchmark_concurrent_processing(self, num_concurrent: int = 3) -> BenchmarkResult:
        """测试并发处理性能"""
        print(f"🔄 开始并发处理性能测试 ({num_concurrent}个并发任务)")

        # 生成测试音频
        audio_data = self.generate_test_audio(3.0)  # 3秒音频

        # 测量开始时的资源
        start_resources = self.measure_system_resources()

        # 开始测试
        start_time = time.time()

        async def process_task(task_id: int):
            """单个处理任务"""
            results = []
            try:
                async for result in cascade.process_audio_file(audio_data):
                    results.append(result)
                return len(results)
            except Exception as e:
                print(f"❌ 任务{task_id}出错: {e}")
                return 0

        # 并发执行任务
        tasks = [process_task(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        test_duration = end_time - start_time

        # 测量结束时的资源
        end_resources = self.measure_system_resources()

        # 计算统计数据
        total_frames = sum(r for r in results if isinstance(r, int))
        success_count = sum(1 for r in results if isinstance(r, int) and r > 0)
        fps = total_frames / test_duration if test_duration > 0 else 0
        success_rate = success_count / num_concurrent

        result = BenchmarkResult(
            test_name=f"并发处理({num_concurrent}任务)",
            duration_seconds=test_duration,
            total_frames=total_frames,
            frames_per_second=fps,
            avg_latency_ms=test_duration * 1000 / num_concurrent,
            max_latency_ms=test_duration * 1000,
            min_latency_ms=test_duration * 1000 / num_concurrent,
            memory_usage_mb=end_resources['memory_mb'],
            cpu_usage_percent=end_resources['cpu_percent'],
            success_rate=success_rate
        )

        self.results.append(result)
        print(f"✅ 并发测试完成: {total_frames}帧, {success_count}/{num_concurrent}任务成功")
        return result

    def print_results(self):
        """打印测试结果"""
        print("\n" + "="*80)
        print("🏆 Cascade 1:1:1架构性能基准测试结果")
        print("="*80)

        for result in self.results:
            print(f"\n📊 {result.test_name}")
            print(f"   ⏱️  测试时长: {result.duration_seconds:.2f}秒")
            print(f"   🎯 处理帧数: {result.total_frames}")
            print(f"   ⚡ 处理速度: {result.frames_per_second:.1f} fps")
            print(f"   📈 平均延迟: {result.avg_latency_ms:.2f}ms")
            print(f"   📊 延迟范围: {result.min_latency_ms:.2f}ms - {result.max_latency_ms:.2f}ms")
            print(f"   💾 内存使用: {result.memory_usage_mb:.1f}MB")
            print(f"   🖥️  CPU使用: {result.cpu_usage_percent:.1f}%")
            print(f"   ✅ 成功率: {result.success_rate*100:.1f}%")

        print("\n" + "="*80)
        print("🎯 性能总结")
        print("="*80)

        if self.results:
            avg_fps = np.mean([r.frames_per_second for r in self.results])
            avg_latency = np.mean([r.avg_latency_ms for r in self.results])
            avg_memory = np.mean([r.memory_usage_mb for r in self.results])
            avg_success = np.mean([r.success_rate for r in self.results])

            print(f"平均处理速度: {avg_fps:.1f} fps")
            print(f"平均处理延迟: {avg_latency:.2f} ms")
            print(f"平均内存使用: {avg_memory:.1f} MB")
            print(f"平均成功率: {avg_success*100:.1f}%")

            # 性能评级
            if avg_fps > 100 and avg_latency < 50:
                grade = "🏆 优秀"
            elif avg_fps > 50 and avg_latency < 100:
                grade = "🥈 良好"
            elif avg_fps > 20 and avg_latency < 200:
                grade = "🥉 一般"
            else:
                grade = "⚠️  需要优化"

            print(f"性能评级: {grade}")


async def main():
    """主测试函数"""
    print("🚀 Cascade 1:1:1架构性能基准测试开始")
    print("="*80)

    benchmark = PerformanceBenchmark()

    try:
        # 1. 单实例处理测试
        await benchmark.benchmark_single_instance(duration=3.0)

        # 2. StreamProcessor测试
        await benchmark.benchmark_stream_processor(duration=3.0)

        # 3. 并发处理测试
        await benchmark.benchmark_concurrent_processing(num_concurrent=2)

        # 打印结果
        benchmark.print_results()

    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
