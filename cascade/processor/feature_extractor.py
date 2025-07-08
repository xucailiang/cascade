"""
特征提取处理器模块

该模块提供了音频特征提取处理器的实现，用于从音频信号中提取各种特征，
如MFCC（梅尔频率倒谱系数）、频谱特征、能量特征等，这些特征可以用于
语音识别、说话人识别、情感分析等任务。
"""

import logging
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, validator

from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
    ProcessorConfig,
)


class FeatureType(str, Enum):
    """特征类型"""
    MFCC = "mfcc"                  # 梅尔频率倒谱系数
    SPECTRAL = "spectral"          # 频谱特征
    ENERGY = "energy"              # 能量特征
    ZERO_CROSSING_RATE = "zcr"     # 过零率
    CHROMA = "chroma"              # 色度特征
    CONTRAST = "contrast"          # 频谱对比度
    TONNETZ = "tonnetz"            # 音调网络特征
    ALL = "all"                    # 所有特征


class FeatureExtractorConfig(ProcessorConfig):
    """特征提取器配置"""
    feature_types: list[FeatureType] = Field(
        default=[FeatureType.MFCC],
        description="要提取的特征类型列表"
    )
    n_mfcc: int = Field(
        default=13,
        description="MFCC特征数量",
        ge=1,
        le=40
    )
    n_fft: int = Field(
        default=2048,
        description="FFT窗口大小",
        ge=512,
        le=4096
    )
    hop_length: int = Field(
        default=512,
        description="帧移（样本数）",
        ge=128,
        le=1024
    )
    n_mels: int = Field(
        default=40,
        description="梅尔滤波器组数量",
        ge=10,
        le=128
    )
    normalize: bool = Field(
        default=True,
        description="是否对特征进行归一化"
    )
    delta: bool = Field(
        default=True,
        description="是否计算一阶差分"
    )
    delta_delta: bool = Field(
        default=False,
        description="是否计算二阶差分"
    )

    @validator('feature_types')
    def validate_feature_types(cls, v):
        """验证特征类型列表"""
        if not v:
            raise ValueError("特征类型列表不能为空")

        # 如果包含ALL，则展开为所有特征类型
        if FeatureType.ALL in v:
            all_types = set(FeatureType)
            all_types.remove(FeatureType.ALL)
            return list(all_types)

        return v

    @validator('n_fft')
    def validate_n_fft(cls, v):
        """验证FFT窗口大小是2的幂"""
        if v & (v - 1) != 0:
            raise ValueError("FFT窗口大小必须是2的幂")
        return v


class FeatureResult(BaseModel):
    """特征提取结果"""
    features: dict[str, np.ndarray] = Field(description="提取的特征字典")
    feature_dims: dict[str, tuple[int, ...]] = Field(description="各特征的维度")
    frame_count: int = Field(description="帧数")
    feature_count: int = Field(description="特征总数")

    class Config:
        arbitrary_types_allowed = True


class FeatureExtractor(AudioProcessor):
    """
    特征提取处理器
    
    从音频信号中提取各种特征，如MFCC、频谱特征、能量特征等。
    """

    def __init__(self, config: FeatureExtractorConfig | None = None):
        """
        初始化特征提取处理器
        
        Args:
            config: 特征提取器配置，如果为None则使用默认配置
        """
        super().__init__(config or FeatureExtractorConfig())
        self.config = config or FeatureExtractorConfig()
        self.logger = logging.getLogger("cascade.processor.feature_extractor")

        # 检查是否有librosa库
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            self.logger.warning("未找到librosa库，某些特征提取功能将不可用。请安装librosa: pip install librosa")
            self.librosa = None

    def process_chunk(self, chunk: AudioChunk) -> FeatureResult:
        """
        处理单个音频块
        
        Args:
            chunk: 音频块
            
        Returns:
            特征提取结果
            
        Raises:
            ValueError: 当音频块无效时
            RuntimeError: 当特征提取失败时
        """
        # 获取音频数据
        audio_data = chunk.data
        sample_rate = chunk.sample_rate

        # 检查音频数据
        if len(audio_data) == 0:
            raise ValueError("音频数据为空")

        # 提取特征
        features = {}
        feature_dims = {}

        # 根据配置提取不同类型的特征
        for feature_type in self.config.feature_types:
            if feature_type == FeatureType.MFCC:
                mfcc, mfcc_dims = self._extract_mfcc(audio_data, sample_rate)
                features["mfcc"] = mfcc
                feature_dims["mfcc"] = mfcc_dims

                # 如果需要计算差分
                if self.config.delta:
                    mfcc_delta, delta_dims = self._compute_delta(mfcc)
                    features["mfcc_delta"] = mfcc_delta
                    feature_dims["mfcc_delta"] = delta_dims

                # 如果需要计算二阶差分
                if self.config.delta_delta:
                    if self.config.delta:
                        mfcc_delta2, delta2_dims = self._compute_delta(mfcc_delta)
                    else:
                        mfcc_delta, _ = self._compute_delta(mfcc)
                        mfcc_delta2, delta2_dims = self._compute_delta(mfcc_delta)

                    features["mfcc_delta2"] = mfcc_delta2
                    feature_dims["mfcc_delta2"] = delta2_dims

            elif feature_type == FeatureType.SPECTRAL:
                spectral, spectral_dims = self._extract_spectral_features(audio_data, sample_rate)
                features.update(spectral)
                feature_dims.update(spectral_dims)

            elif feature_type == FeatureType.ENERGY:
                energy, energy_dims = self._extract_energy_features(audio_data, sample_rate)
                features.update(energy)
                feature_dims.update(energy_dims)

            elif feature_type == FeatureType.ZERO_CROSSING_RATE:
                zcr, zcr_dims = self._extract_zero_crossing_rate(audio_data, sample_rate)
                features["zcr"] = zcr
                feature_dims["zcr"] = zcr_dims

            elif feature_type == FeatureType.CHROMA:
                chroma, chroma_dims = self._extract_chroma_features(audio_data, sample_rate)
                features["chroma"] = chroma
                feature_dims["chroma"] = chroma_dims

            elif feature_type == FeatureType.CONTRAST:
                contrast, contrast_dims = self._extract_contrast_features(audio_data, sample_rate)
                features["contrast"] = contrast
                feature_dims["contrast"] = contrast_dims

            elif feature_type == FeatureType.TONNETZ:
                tonnetz, tonnetz_dims = self._extract_tonnetz_features(audio_data, sample_rate)
                features["tonnetz"] = tonnetz
                feature_dims["tonnetz"] = tonnetz_dims

        # 计算总帧数和特征数
        frame_count = next(iter(features.values())).shape[0] if features else 0
        feature_count = sum(np.prod(dims) for dims in feature_dims.values())

        # 创建结果
        result = FeatureResult(
            features=features,
            feature_dims=feature_dims,
            frame_count=frame_count,
            feature_count=feature_count
        )

        return result

    def _extract_mfcc(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        提取MFCC特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            MFCC特征和其维度
            
        Raises:
            RuntimeError: 当特征提取失败时
        """
        if self.librosa is None:
            # 使用简单的实现
            return self._extract_mfcc_simple(audio_data, sample_rate)

        try:
            # 使用librosa提取MFCC
            mfcc = self.librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels
            )

            # 转置为(帧数, 特征数)的形状
            mfcc = mfcc.T

            # 归一化
            if self.config.normalize:
                mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10)

            return mfcc, mfcc.shape

        except Exception as e:
            self.logger.error(f"MFCC特征提取失败: {str(e)}")
            raise RuntimeError(f"MFCC特征提取失败: {str(e)}")

    def _extract_mfcc_simple(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        使用简单方法提取MFCC特征（不依赖librosa）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            MFCC特征和其维度
        """
        # 计算帧数
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        n_frames = 1 + (len(audio_data) - frame_length) // hop_length

        # 创建帧
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_data):
                frames[i] = audio_data[start:end] * np.hanning(frame_length)

        # 计算功率谱
        magnitude_spectra = np.abs(np.fft.rfft(frames, n=self.config.n_fft))
        power_spectra = magnitude_spectra ** 2

        # 创建梅尔滤波器组
        n_mels = self.config.n_mels
        n_fft = self.config.n_fft
        fmin, fmax = 0.0, sample_rate / 2.0

        # 梅尔刻度转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595.0) - 1)

        # 创建梅尔滤波器组
        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        mel_filters = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            start, center, end = bin_indices[i:i+3]
            # 创建三角滤波器
            mel_filters[i, start:center] = np.linspace(0, 1, center - start)
            mel_filters[i, center:end] = np.linspace(1, 0, end - center)

        # 应用梅尔滤波器
        mel_spectra = np.dot(power_spectra, mel_filters.T)

        # 取对数
        log_mel_spectra = np.log(mel_spectra + 1e-10)

        # 应用离散余弦变换
        n_mfcc = min(self.config.n_mfcc, n_mels)
        dct_matrix = np.zeros((n_mfcc, n_mels))
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_mels)

        mfcc = np.dot(log_mel_spectra, dct_matrix.T)

        # 归一化
        if self.config.normalize:
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10)

        return mfcc, mfcc.shape

    def _compute_delta(self, features: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        计算特征的差分
        
        Args:
            features: 特征矩阵
            
        Returns:
            差分特征和其维度
        """
        # 简单的差分计算
        padded = np.pad(features, ((1, 1), (0, 0)), mode='edge')
        delta = (padded[2:] - padded[:-2]) / 2.0

        return delta, delta.shape

    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
        """
        提取频谱特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            频谱特征字典和维度字典
        """
        if self.librosa is None:
            # 返回简单的频谱特征
            return self._extract_spectral_features_simple(audio_data, sample_rate)

        try:
            features = {}
            feature_dims = {}

            # 频谱质心
            centroid = self.librosa.feature.spectral_centroid(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features["spectral_centroid"] = centroid.T
            feature_dims["spectral_centroid"] = centroid.T.shape

            # 频谱带宽
            bandwidth = self.librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features["spectral_bandwidth"] = bandwidth.T
            feature_dims["spectral_bandwidth"] = bandwidth.T.shape

            # 频谱平坦度
            flatness = self.librosa.feature.spectral_flatness(
                y=audio_data,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features["spectral_flatness"] = flatness.T
            feature_dims["spectral_flatness"] = flatness.T.shape

            # 频谱滚降
            rolloff = self.librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features["spectral_rolloff"] = rolloff.T
            feature_dims["spectral_rolloff"] = rolloff.T.shape

            return features, feature_dims

        except Exception as e:
            self.logger.error(f"频谱特征提取失败: {str(e)}")
            return {}, {}

    def _extract_spectral_features_simple(self, audio_data: np.ndarray, sample_rate: int) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
        """
        使用简单方法提取频谱特征（不依赖librosa）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            频谱特征字典和维度字典
        """
        features = {}
        feature_dims = {}

        # 计算帧数
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        n_frames = 1 + (len(audio_data) - frame_length) // hop_length

        # 创建帧
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_data):
                frames[i] = audio_data[start:end] * np.hanning(frame_length)

        # 计算功率谱
        magnitude_spectra = np.abs(np.fft.rfft(frames, n=self.config.n_fft))
        power_spectra = magnitude_spectra ** 2

        # 频率轴
        freqs = np.fft.rfftfreq(self.config.n_fft, 1.0 / sample_rate)

        # 频谱质心
        spectral_centroid = np.zeros(n_frames)
        for i in range(n_frames):
            if np.sum(power_spectra[i]) > 0:
                spectral_centroid[i] = np.sum(freqs * power_spectra[i]) / np.sum(power_spectra[i])

        features["spectral_centroid"] = spectral_centroid.reshape(-1, 1)
        feature_dims["spectral_centroid"] = features["spectral_centroid"].shape

        return features, feature_dims

    def _extract_energy_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
        """
        提取能量特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            能量特征字典和维度字典
        """
        features = {}
        feature_dims = {}

        # 计算帧数
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        n_frames = 1 + (len(audio_data) - frame_length) // hop_length

        # 创建帧
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_data):
                frames[i] = audio_data[start:end]

        # 计算短时能量
        energy = np.sum(frames ** 2, axis=1)

        # 计算对数能量
        log_energy = np.log(energy + 1e-10)

        features["energy"] = energy.reshape(-1, 1)
        feature_dims["energy"] = features["energy"].shape

        features["log_energy"] = log_energy.reshape(-1, 1)
        feature_dims["log_energy"] = features["log_energy"].shape

        return features, feature_dims

    def _extract_zero_crossing_rate(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        提取过零率特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            过零率特征和其维度
        """
        if self.librosa is None:
            # 使用简单的实现
            return self._extract_zero_crossing_rate_simple(audio_data, sample_rate)

        try:
            zcr = self.librosa.feature.zero_crossing_rate(
                y=audio_data,
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length
            )

            return zcr.T, zcr.T.shape

        except Exception as e:
            self.logger.error(f"过零率特征提取失败: {str(e)}")
            return np.array([]), (0,)

    def _extract_zero_crossing_rate_simple(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        使用简单方法提取过零率特征（不依赖librosa）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            过零率特征和其维度
        """
        # 计算帧数
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        n_frames = 1 + (len(audio_data) - frame_length) // hop_length

        # 创建帧
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_data):
                frames[i] = audio_data[start:end]

        # 计算过零率
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            zcr[i] = np.sum(np.abs(np.diff(np.signbit(frames[i])))) / (2 * frame_length)

        return zcr.reshape(-1, 1), zcr.reshape(-1, 1).shape

    def _extract_chroma_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        提取色度特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            色度特征和其维度
        """
        if self.librosa is None:
            # 返回空特征
            return np.array([]), (0,)

        try:
            chroma = self.librosa.feature.chroma_stft(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )

            return chroma.T, chroma.T.shape

        except Exception as e:
            self.logger.error(f"色度特征提取失败: {str(e)}")
            return np.array([]), (0,)

    def _extract_contrast_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        提取频谱对比度特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            频谱对比度特征和其维度
        """
        if self.librosa is None:
            # 返回空特征
            return np.array([]), (0,)

        try:
            contrast = self.librosa.feature.spectral_contrast(
                y=audio_data,
                sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )

            return contrast.T, contrast.T.shape

        except Exception as e:
            self.logger.error(f"频谱对比度特征提取失败: {str(e)}")
            return np.array([]), (0,)

    def _extract_tonnetz_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        提取音调网络特征
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            音调网络特征和其维度
        """
        if self.librosa is None:
            # 返回空特征
            return np.array([]), (0,)

        try:
            tonnetz = self.librosa.feature.tonnetz(
                y=audio_data,
                sr=sample_rate
            )

            return tonnetz.T, tonnetz.T.shape

        except Exception as e:
            self.logger.error(f"音调网络特征提取失败: {str(e)}")
            return np.array([]), (0,)
