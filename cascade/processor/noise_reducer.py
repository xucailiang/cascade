"""
噪声抑制处理器模块

该模块提供了噪声抑制处理器的实现，用于减少或消除音频中的背景噪声，提高语音的清晰度。
支持多种噪声抑制算法，包括频谱减法、维纳滤波等。
"""

import logging
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, field_validator

from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
    ProcessorConfig,
)


class NoiseReductionMethod(str, Enum):
    """噪声抑制方法"""
    SPECTRAL_SUBTRACTION = "spectral_subtraction"  # 频谱减法
    WIENER_FILTER = "wiener_filter"                # 维纳滤波
    MINIMUM_MEAN_SQUARE = "minimum_mean_square"    # 最小均方误差


class NoiseReducerConfig(ProcessorConfig):
    """噪声抑制器配置"""
    method: NoiseReductionMethod = Field(
        default=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
        description="噪声抑制方法"
    )
    noise_threshold: float = Field(
        default=0.05,
        description="噪声阈值，低于此值的信号被视为噪声",
        ge=0.0,
        le=1.0
    )
    reduction_factor: float = Field(
        default=0.5,
        description="噪声抑制因子，值越大抑制越强",
        ge=0.0,
        le=1.0
    )
    smoothing_factor: float = Field(
        default=0.8,
        description="平滑因子，用于平滑噪声估计",
        ge=0.0,
        le=1.0
    )
    fft_size: int = Field(
        default=512,
        description="FFT大小",
        ge=128,
        le=4096
    )
    noise_estimation_frames: int = Field(
        default=10,
        description="用于噪声估计的帧数",
        ge=1,
        le=100
    )

    @field_validator('fft_size')
    def validate_fft_size(cls, v):
        """验证FFT大小是2的幂"""
        if v & (v - 1) != 0:
            raise ValueError("FFT大小必须是2的幂")
        return v


class NoiseReductionResult(BaseModel):
    """噪声抑制结果"""
    original_energy: float = Field(description="原始信号能量")
    reduced_energy: float = Field(description="抑制后信号能量")
    noise_energy: float = Field(description="估计的噪声能量")
    reduction_ratio: float = Field(description="噪声抑制比例")
    snr_before: float = Field(description="处理前信噪比")
    snr_after: float = Field(description="处理后信噪比")

    class Config:
        arbitrary_types_allowed = True


class NoiseReducer(AudioProcessor):
    """
    噪声抑制处理器
    
    使用频谱减法或维纳滤波等方法减少音频中的背景噪声。
    """

    def __init__(self, config: NoiseReducerConfig | None = None):
        """
        初始化噪声抑制处理器
        
        Args:
            config: 噪声抑制器配置，如果为None则使用默认配置
        """
        super().__init__(config or NoiseReducerConfig())
        self.config = config or NoiseReducerConfig()
        self.logger = logging.getLogger("cascade.processor.noise_reducer")

        # 噪声估计
        self.noise_profile = None
        self.frame_count = 0

    def process_chunk(self, chunk: AudioChunk) -> NoiseReductionResult:
        """
        处理单个音频块
        
        Args:
            chunk: 音频块
            
        Returns:
            噪声抑制结果
            
        Raises:
            ValueError: 当音频块无效时
        """
        # 获取音频数据
        audio_data = chunk.data

        # 计算原始信号能量
        original_energy = np.mean(audio_data ** 2)

        # 根据选择的方法进行噪声抑制
        if self.config.method == NoiseReductionMethod.SPECTRAL_SUBTRACTION:
            processed_data, noise_energy = self._spectral_subtraction(audio_data)
        elif self.config.method == NoiseReductionMethod.WIENER_FILTER:
            processed_data, noise_energy = self._wiener_filter(audio_data)
        elif self.config.method == NoiseReductionMethod.MINIMUM_MEAN_SQUARE:
            processed_data, noise_energy = self._minimum_mean_square(audio_data)
        else:
            # 默认使用频谱减法
            processed_data, noise_energy = self._spectral_subtraction(audio_data)

        # 计算处理后信号能量
        reduced_energy = np.mean(processed_data ** 2)

        # 计算信噪比
        snr_before = 10 * np.log10(original_energy / noise_energy) if noise_energy > 0 else 100.0
        snr_after = 10 * np.log10(reduced_energy / noise_energy) if noise_energy > 0 else 100.0

        # 计算噪声抑制比例
        reduction_ratio = 1.0 - (reduced_energy / original_energy) if original_energy > 0 else 0.0

        # 更新音频块数据
        chunk.data = processed_data

        # 创建结果
        result = NoiseReductionResult(
            original_energy=float(original_energy),
            reduced_energy=float(reduced_energy),
            noise_energy=float(noise_energy),
            reduction_ratio=float(reduction_ratio),
            snr_before=float(snr_before),
            snr_after=float(snr_after)
        )

        return result

    def _spectral_subtraction(self, audio_data: np.ndarray) -> tuple[np.ndarray, float]:
        """
        使用频谱减法进行噪声抑制
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理后的音频数据和估计的噪声能量
        """
        # 获取配置参数
        fft_size = self.config.fft_size
        reduction_factor = self.config.reduction_factor
        smoothing_factor = self.config.smoothing_factor

        # 应用汉宁窗
        windowed_data = audio_data * np.hanning(len(audio_data))

        # 计算FFT
        spectrum = np.fft.rfft(windowed_data, n=fft_size)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # 估计噪声谱
        if self.noise_profile is None:
            # 初始化噪声谱
            self.noise_profile = magnitude
            self.frame_count = 1
        elif self.frame_count < self.config.noise_estimation_frames:
            # 更新噪声谱
            self.noise_profile = smoothing_factor * self.noise_profile + (1 - smoothing_factor) * magnitude
            self.frame_count += 1

        # 计算噪声能量
        noise_energy = np.mean(self.noise_profile ** 2)

        # 频谱减法
        subtracted_magnitude = np.maximum(magnitude - reduction_factor * self.noise_profile, 0.01 * magnitude)

        # 重建信号
        processed_spectrum = subtracted_magnitude * np.exp(1j * phase)
        processed_data = np.fft.irfft(processed_spectrum, n=fft_size)

        # 截取与原始数据相同长度
        processed_data = processed_data[:len(audio_data)]

        return processed_data, float(noise_energy)

    def _wiener_filter(self, audio_data: np.ndarray) -> tuple[np.ndarray, float]:
        """
        使用维纳滤波进行噪声抑制
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理后的音频数据和估计的噪声能量
        """
        # 获取配置参数
        fft_size = self.config.fft_size
        smoothing_factor = self.config.smoothing_factor

        # 应用汉宁窗
        windowed_data = audio_data * np.hanning(len(audio_data))

        # 计算FFT
        spectrum = np.fft.rfft(windowed_data, n=fft_size)
        magnitude_squared = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)

        # 估计噪声谱
        if self.noise_profile is None:
            # 初始化噪声谱
            self.noise_profile = magnitude_squared
            self.frame_count = 1
        elif self.frame_count < self.config.noise_estimation_frames:
            # 更新噪声谱
            self.noise_profile = smoothing_factor * self.noise_profile + (1 - smoothing_factor) * magnitude_squared
            self.frame_count += 1

        # 计算噪声能量
        noise_energy = np.mean(self.noise_profile)

        # 计算维纳滤波器增益
        # G(f) = |S(f)|^2 / (|S(f)|^2 + |N(f)|^2)
        gain = magnitude_squared / (magnitude_squared + self.noise_profile)

        # 应用增益
        processed_spectrum = np.sqrt(gain * magnitude_squared) * np.exp(1j * phase)

        # 重建信号
        processed_data = np.fft.irfft(processed_spectrum, n=fft_size)

        # 截取与原始数据相同长度
        processed_data = processed_data[:len(audio_data)]

        return processed_data, float(noise_energy)

    def _minimum_mean_square(self, audio_data: np.ndarray) -> tuple[np.ndarray, float]:
        """
        使用最小均方误差方法进行噪声抑制
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理后的音频数据和估计的噪声能量
        """
        # 获取配置参数
        fft_size = self.config.fft_size
        smoothing_factor = self.config.smoothing_factor

        # 应用汉宁窗
        windowed_data = audio_data * np.hanning(len(audio_data))

        # 计算FFT
        spectrum = np.fft.rfft(windowed_data, n=fft_size)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # 估计噪声谱
        if self.noise_profile is None:
            # 初始化噪声谱
            self.noise_profile = magnitude
            self.frame_count = 1
        elif self.frame_count < self.config.noise_estimation_frames:
            # 更新噪声谱
            self.noise_profile = smoothing_factor * self.noise_profile + (1 - smoothing_factor) * magnitude
            self.frame_count += 1

        # 计算噪声能量
        noise_energy = np.mean(self.noise_profile ** 2)

        # 计算先验信噪比
        prior_snr = np.maximum(magnitude ** 2 / (self.noise_profile ** 2) - 1, 0)

        # 计算MMSE增益
        gain = prior_snr / (1 + prior_snr)

        # 应用增益
        processed_magnitude = gain * magnitude
        processed_spectrum = processed_magnitude * np.exp(1j * phase)

        # 重建信号
        processed_data = np.fft.irfft(processed_spectrum, n=fft_size)

        # 截取与原始数据相同长度
        processed_data = processed_data[:len(audio_data)]

        return processed_data, float(noise_energy)

    def reset(self) -> None:
        """重置噪声估计"""
        self.noise_profile = None
        self.frame_count = 0
