import wave
import numpy as np
import os

def create_silent_wav(file_path: str, duration_s: int = 2, sample_rate: int = 16000):
    """
    创建一个静音的WAV文件
    """
    num_samples = duration_s * sample_rate
    # 创建静音数据
    silent_data = np.zeros(num_samples, dtype=np.int16)

    try:
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(silent_data.tobytes())
        print(f"成功创建测试文件: {file_path}")
    except Exception as e:
        print(f"创建测试文件失败: {e}")

if __name__ == "__main__":
    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, "silent_16k_s16le.wav")
    create_silent_wav(file_path)