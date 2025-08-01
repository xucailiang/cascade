#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD时间戳准确性测试脚本

专注于验证Cascade VAD系统的时间戳准确性和语音块拼接效果。
使用Ground Truth数据: 0.768330秒 - 5.009294秒

功能：
1. 加载音频文件和Ground Truth标注
2. 运行VAD检测并收集结果
3. 分析时间戳准确性
4. 生成可视化测试报告

使用方法：
    python tests/vad_timestamp_accuracy_test.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pydantic import BaseModel

# Cascade导入
from cascade.types import VADConfig, AudioConfig, AudioChunk, VADResult
from cascade.backends import create_vad_backend

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestConfig(BaseModel):
    """测试配置"""
    audio_file: str = "请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    ground_truth_file: str = "ground_truth.txt"
    output_report: str = "vad_timestamp_accuracy_report.png"
    
    # VAD配置
    vad_threshold: float = 0.5
    chunk_duration_ms: int = 512
    overlap_ms: int = 50
    
    # 准确性阈值
    acceptable_error_ms: float = 100.0  # 可接受的时间戳误差


class GroundTruthSegment(BaseModel):
    """Ground Truth语音段"""
    start_sec: float
    end_sec: float
    duration_sec: float
    
    @classmethod
    def from_audacity_line(cls, line: str) -> Optional['GroundTruthSegment']:
        """从Audacity导出的标签行创建实例"""
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                start = float(parts[0])
                end = float(parts[1])
                return cls(
                    start_sec=start,
                    end_sec=end,
                    duration_sec=end - start
                )
            except ValueError:
                return None
        return None


class VADDetectionResult(BaseModel):
    """VAD检测结果"""
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    chunk_count: int


class AccuracyMetrics(BaseModel):
    """准确性指标"""
    start_error_ms: float
    end_error_ms: float
    average_error_ms: float
    duration_error_sec: float
    is_detected: bool
    detection_count: int
    
    # 评估结果
    start_accuracy: str  # "优秀"/"良好"/"可接受"/"不合格"
    end_accuracy: str
    overall_accuracy: str


class VADTimestampAccuracyTest:
    """VAD时间戳准确性测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.ground_truth: Optional[GroundTruthSegment] = None
        self.vad_results: List[VADResult] = []
        self.detected_segments: List[VADDetectionResult] = []
        self.metrics: Optional[AccuracyMetrics] = None
        
    def load_ground_truth(self) -> bool:
        """加载Ground Truth数据"""
        try:
            gt_path = Path(self.config.ground_truth_file)
            if not gt_path.exists():
                logger.error(f"Ground Truth文件不存在: {gt_path}")
                return False
                
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    segment = GroundTruthSegment.from_audacity_line(line)
                    if segment:
                        self.ground_truth = segment
                        logger.info(f"加载Ground Truth: {segment.start_sec:.3f}s - {segment.end_sec:.3f}s")
                        return True
                        
            logger.error("未找到有效的Ground Truth数据")
            return False
            
        except Exception as e:
            logger.error(f"加载Ground Truth失败: {e}")
            return False
    
    def load_audio(self) -> Optional[np.ndarray]:
        """加载音频文件"""
        try:
            audio_path = Path(self.config.audio_file)
            if not audio_path.exists():
                logger.error(f"音频文件不存在: {audio_path}")
                return None
                
            # 使用soundfile加载音频
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
            
            logger.info(f"音频加载成功: {len(audio_data)/sample_rate:.2f}秒, {sample_rate}Hz")
            
            # 如果是立体声，转换为单声道
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # 重采样到16kHz（如果需要）
            if sample_rate != 16000:
                from scipy import signal
                audio_data = signal.resample(
                    audio_data, 
                    int(len(audio_data) * 16000 / sample_rate)
                )
                logger.info(f"音频重采样: {sample_rate}Hz -> 16000Hz")
                
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"音频加载失败: {e}")
            return None
    
    async def run_vad_detection(self, audio_data: np.ndarray) -> bool:
        """运行VAD检测"""
        try:
            # 配置VAD (启用100ms延迟补偿)
            vad_config = VADConfig(
                backend="silero",
                threshold=self.config.vad_threshold,
                chunk_duration_ms=self.config.chunk_duration_ms,
                overlap_ms=self.config.overlap_ms,
                min_speech_duration_ms=100,
                workers=1,
                compensation_ms=250  # 启用100ms延迟补偿
            )
            
            audio_config = AudioConfig(
                sample_rate=16000,
                channels=1,
                format="wav",
                dtype="float32"
            )
            
            logger.info("初始化Silero VAD后端...")
            backend = create_vad_backend(vad_config)
            await backend.initialize()
            
            # 分块处理音频
            chunk_size = int(self.config.chunk_duration_ms * 16000 / 1000)  # 512ms @ 16kHz
            overlap_size = int(self.config.overlap_ms * 16000 / 1000)      # 50ms @ 16kHz
            step_size = chunk_size - overlap_size
            
            logger.info(f"开始VAD检测: 块大小={chunk_size}, 重叠={overlap_size}, 步长={step_size}")
            
            self.vad_results = []
            for i in range(0, len(audio_data), step_size):
                # 提取音频块
                chunk_data = audio_data[i:i+chunk_size]
                if len(chunk_data) < chunk_size:
                    # 最后一块补零
                    padded = np.zeros(chunk_size, dtype=np.float32)
                    padded[:len(chunk_data)] = chunk_data
                    chunk_data = padded
                
                # 创建AudioChunk
                timestamp_ms = i / 16000 * 1000
                chunk = AudioChunk(
                    data=chunk_data,
                    sequence_number=len(self.vad_results),
                    start_frame=i,
                    chunk_size=chunk_size,
                    timestamp_ms=timestamp_ms,
                    sample_rate=16000
                )
                
                # VAD检测
                result = backend.process_chunk(chunk)
                self.vad_results.append(result)
                
                # 实时日志
                time_str = f"{result.start_ms/1000:.2f}-{result.end_ms/1000:.2f}s"
                status = "🗣️语音" if result.is_speech else "🔇静音"
                logger.debug(f"{status} | {time_str} | 概率: {result.probability:.3f}")
            
            await backend.close()
            logger.info(f"VAD检测完成: {len(self.vad_results)}个块")
            return True
            
        except Exception as e:
            logger.error(f"VAD检测失败: {e}")
            return False
    
    def merge_speech_segments(self) -> None:
        """合并连续的语音块为语音段"""
        if not self.vad_results:
            return
            
        current_segment = None
        speech_chunks = []
        
        for result in self.vad_results:
            if result.is_speech:
                if current_segment is None:
                    # 开始新语音段
                    current_segment = {
                        'start_ms': result.start_ms,
                        'end_ms': result.end_ms,
                        'confidences': [result.confidence],
                        'chunk_count': 1
                    }
                else:
                    # 扩展当前语音段
                    current_segment['end_ms'] = result.end_ms
                    current_segment['confidences'].append(result.confidence)
                    current_segment['chunk_count'] += 1
            else:
                if current_segment is not None:
                    # 语音段结束
                    speech_chunks.append(current_segment)
                    current_segment = None
        
        # 处理最后一个语音段
        if current_segment is not None:
            speech_chunks.append(current_segment)
        
        # 转换为VADDetectionResult
        self.detected_segments = []
        for segment in speech_chunks:
            self.detected_segments.append(VADDetectionResult(
                start_sec=segment['start_ms'] / 1000.0,
                end_sec=segment['end_ms'] / 1000.0,
                duration_sec=(segment['end_ms'] - segment['start_ms']) / 1000.0,
                confidence=np.mean(segment['confidences']),
                chunk_count=segment['chunk_count']
            ))
        
        logger.info(f"检测到 {len(self.detected_segments)} 个语音段")
        for i, segment in enumerate(self.detected_segments):
            logger.info(f"  语音段{i+1}: {segment.start_sec:.3f}s-{segment.end_sec:.3f}s "
                       f"(时长: {segment.duration_sec:.3f}s, 置信度: {segment.confidence:.3f})")
    
    def calculate_accuracy_metrics(self) -> None:
        """计算准确性指标"""
        if not self.ground_truth:
            logger.error("缺少Ground Truth数据")
            return
            
        # 默认指标（未检测到语音）
        start_error = float('inf')
        end_error = float('inf')
        is_detected = len(self.detected_segments) > 0
        
        if is_detected and len(self.detected_segments) > 0:
            # 使用第一个检测到的语音段（理想情况下应该只有一个）
            primary_segment = self.detected_segments[0]
            
            # 如果有多个语音段，合并为一个连续段
            if len(self.detected_segments) > 1:
                start_sec = min(s.start_sec for s in self.detected_segments)
                end_sec = max(s.end_sec for s in self.detected_segments)
                primary_segment = VADDetectionResult(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                    confidence=np.mean([s.confidence for s in self.detected_segments]),
                    chunk_count=sum(s.chunk_count for s in self.detected_segments)
                )
            
            # 计算时间戳误差
            start_error = abs(primary_segment.start_sec - self.ground_truth.start_sec) * 1000  # ms
            end_error = abs(primary_segment.end_sec - self.ground_truth.end_sec) * 1000        # ms
        
        average_error = (start_error + end_error) / 2 if start_error != float('inf') else float('inf')
        
        # 计算时长误差
        if is_detected and len(self.detected_segments) > 0:
            detected_duration = sum(s.duration_sec for s in self.detected_segments)
            duration_error = abs(detected_duration - self.ground_truth.duration_sec)
        else:
            duration_error = self.ground_truth.duration_sec
        
        # 评估准确性等级
        def evaluate_accuracy(error_ms: float) -> str:
            if error_ms == float('inf'):
                return "不合格"
            elif error_ms <= 50:
                return "优秀"
            elif error_ms <= 100:
                return "良好"
            elif error_ms <= 200:
                return "可接受"
            else:
                return "不合格"
        
        self.metrics = AccuracyMetrics(
            start_error_ms=start_error,
            end_error_ms=end_error,
            average_error_ms=average_error,
            duration_error_sec=duration_error,
            is_detected=is_detected,
            detection_count=len(self.detected_segments),
            start_accuracy=evaluate_accuracy(start_error),
            end_accuracy=evaluate_accuracy(end_error),
            overall_accuracy=evaluate_accuracy(average_error)
        )
    
    def generate_visualization(self, audio_data: np.ndarray) -> None:
        """生成可视化测试报告"""
        if not self.ground_truth or not self.metrics:
            logger.error("缺少必要数据，无法生成可视化报告")
            return
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            
            # 时间轴
            time_axis = np.arange(len(audio_data)) / 16000
            
            # === 第一个子图：音频波形和时间戳对比 ===
            ax1.plot(time_axis, audio_data, color='gray', alpha=0.7, linewidth=0.5, label='音频波形')
            
            # Ground Truth区域（绿色）
            ax1.axvspan(self.ground_truth.start_sec, self.ground_truth.end_sec, 
                       color='green', alpha=0.3, label='Ground Truth')
            
            # VAD检测区域（红色）
            for segment in self.detected_segments:
                ax1.axvspan(segment.start_sec, segment.end_sec, 
                           color='red', alpha=0.4, label='VAD检测' if segment == self.detected_segments[0] else "")
            
            # 标注时间戳
            ax1.axvline(self.ground_truth.start_sec, color='green', linestyle='--', alpha=0.8)
            ax1.axvline(self.ground_truth.end_sec, color='green', linestyle='--', alpha=0.8)
            
            if self.detected_segments:
                primary = self.detected_segments[0]
                ax1.axvline(primary.start_sec, color='red', linestyle='--', alpha=0.8)
                ax1.axvline(primary.end_sec, color='red', linestyle='--', alpha=0.8)
            
            ax1.set_title('VAD时间戳准确性测试报告', fontsize=16, fontweight='bold')
            ax1.set_xlabel('时间 (秒)', fontsize=12)
            ax1.set_ylabel('振幅', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # === 第二个子图：VAD概率曲线 ===
            if self.vad_results:
                times = [r.start_ms/1000 for r in self.vad_results]
                probs = [r.probability for r in self.vad_results]
                
                ax2.plot(times, probs, 'r-', linewidth=2, label='VAD概率')
                ax2.axhline(y=self.config.vad_threshold, color='blue', linestyle='--', 
                           alpha=0.7, label=f'阈值 ({self.config.vad_threshold})')
                
                # 标注语音区域
                for result in self.vad_results:
                    if result.is_speech:
                        ax2.axvspan(result.start_ms/1000, result.end_ms/1000, 
                                   color='yellow', alpha=0.2)
                
                ax2.set_xlabel('时间 (秒)', fontsize=12)
                ax2.set_ylabel('VAD概率', fontsize=12)
                ax2.set_ylim(0, 1)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # === 添加统计信息 ===
            stats_text = self._generate_stats_text()
            fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.25, 1, 1])
            plt.savefig(self.config.output_report, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化报告已生成: {self.config.output_report}")
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")
    
    def _generate_stats_text(self) -> str:
        """生成统计信息文本"""
        if not self.metrics or not self.ground_truth:
            return ""
        
        stats = [
            "=== VAD时间戳准确性测试结果 ===",
            f"Ground Truth: {self.ground_truth.start_sec:.3f}s - {self.ground_truth.end_sec:.3f}s",
            f"检测结果: {len(self.detected_segments)}个语音段",
            "",
            "=== 时间戳误差分析 ===",
            f"开始时间误差: {self.metrics.start_error_ms:.1f}ms ({self.metrics.start_accuracy})",
            f"结束时间误差: {self.metrics.end_error_ms:.1f}ms ({self.metrics.end_accuracy})",
            f"平均时间戳误差: {self.metrics.average_error_ms:.1f}ms ({self.metrics.overall_accuracy})",
            f"时长误差: {self.metrics.duration_error_sec:.3f}s",
            "",
            "=== 检测状态 ===",
            f"是否检测到语音: {'是' if self.metrics.is_detected else '否'}",
            f"语音段数量: {self.metrics.detection_count}",
            f"VAD配置: 阈值={self.config.vad_threshold}, 块大小={self.config.chunk_duration_ms}ms",
        ]
        
        if self.detected_segments:
            stats.extend([
                "",
                "=== 检测到的语音段 ===",
            ])
            for i, segment in enumerate(self.detected_segments):
                stats.append(f"段{i+1}: {segment.start_sec:.3f}s-{segment.end_sec:.3f}s "
                           f"(时长:{segment.duration_sec:.3f}s, 置信度:{segment.confidence:.3f})")
        
        return "\n".join(stats)
    
    async def run_test(self) -> bool:
        """执行完整测试流程"""
        logger.info("🚀 开始VAD时间戳准确性测试")
        
        # 1. 加载Ground Truth
        if not self.load_ground_truth():
            return False
        
        # 2. 加载音频
        audio_data = self.load_audio()
        if audio_data is None:
            return False
        
        # 3. VAD检测
        if not await self.run_vad_detection(audio_data):
            return False
        
        # 4. 合并语音段
        self.merge_speech_segments()
        
        # 5. 计算准确性指标
        self.calculate_accuracy_metrics()
        
        # 6. 生成可视化报告
        self.generate_visualization(audio_data)
        
        # 7. 输出结果
        self._print_test_results()
        
        logger.info("✅ VAD时间戳准确性测试完成")
        return True
    
    def _print_test_results(self) -> None:
        """打印测试结果"""
        if not self.metrics:
            return
        
        print("\n" + "="*60)
        print("🎯 VAD时间戳准确性测试结果")
        print("="*60)
        
        print(f"📊 总体评估: {self.metrics.overall_accuracy}")
        print(f"📍 平均时间戳误差: {self.metrics.average_error_ms:.1f}ms")
        print(f"🎤 检测语音段数: {self.metrics.detection_count}")
        
        if self.metrics.is_detected:
            print(f"✅ 检测状态: 成功检测到语音")
            print(f"⏱️  开始时间误差: {self.metrics.start_error_ms:.1f}ms ({self.metrics.start_accuracy})")
            print(f"⏱️  结束时间误差: {self.metrics.end_error_ms:.1f}ms ({self.metrics.end_accuracy})")
        else:
            print(f"❌ 检测状态: 未检测到语音")
        
        print(f"📈 可视化报告: {self.config.output_report}")
        print("="*60)


async def main():
    """主函数"""
    print("🎬 VAD时间戳准确性测试工具")
    print("专用于验证Cascade VAD系统的时间戳准确性")
    print("-" * 50)
    
    # 创建测试配置
    config = TestConfig()
    
    # 检查文件存在性
    if not Path(config.audio_file).exists():
        print(f"❌ 音频文件不存在: {config.audio_file}")
        print("请将测试音频文件放在当前目录下")
        return
    
    if not Path(config.ground_truth_file).exists():
        print(f"❌ Ground Truth文件不存在: {config.ground_truth_file}")
        print("请参考docs/how_to_label_ground_truth.md创建标注文件")
        return
    
    # 运行测试
    test = VADTimestampAccuracyTest(config)
    success = await test.run_test()
    
    if success:
        print("\n🎉 测试执行成功！")
        print(f"📊 查看详细报告: {config.output_report}")
    else:
        print("\n❌ 测试执行失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())