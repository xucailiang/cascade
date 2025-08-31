import React, { useState, memo } from 'react';

/**
 * 语音段列表组件
 * 
 * @param {Object} props - 组件属性
 * @param {Array} props.segments - 语音段数组
 * @param {number} props.sampleRate - 音频采样率
 * @returns {JSX.Element} 语音段列表组件
 */
const SpeechSegmentListComponent = ({ segments = [], sampleRate }) => {
  const [selectedSegment, setSelectedSegment] = useState(null);
  const audioContextRef = React.useRef(new (window.AudioContext || window.webkitAudioContext)());
  
  // 播放语音段
  const playSegment = async (segment) => {
    if (!segment || !segment.audio_data) {
      console.error('语音段没有音频数据');
      return;
    }
    if (selectedSegment) return; // 防止重复点击

    try {
      setSelectedSegment(segment.segment_id);

      // 1. Base64解码
      const binaryString = atob(segment.audio_data);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // 2. 将16-bit PCM (Int16Array) 转换为 32-bit Float
      const pcmData = new Int16Array(bytes.buffer);
      const float32Data = new Float32Array(pcmData.length);
      for (let i = 0; i < pcmData.length; i++) {
        float32Data[i] = pcmData[i] / 32768.0; // 归一化到 -1.0 到 1.0
      }

      // 3. 创建AudioBuffer并填充数据
      const audioContext = audioContextRef.current;
      const buffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
      buffer.copyToChannel(float32Data, 0);

      // 4. 播放
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.start(0);

      source.onended = () => {
        setSelectedSegment(null);
      };
    } catch (error) {
      console.error('播放语音段失败', error);
      setSelectedSegment(null);
    }
  };
  
  // 格式化时间戳
  const formatTimestamp = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const milliseconds = Math.floor(ms % 1000);
    return `${seconds}.${milliseconds.toString().padStart(3, '0')}s`;
  };
  
  // 如果没有语音段，显示提示信息
  if (segments.length === 0) {
    return (
      <div className="card">
        <h3 className="text-xl font-bold mb-4">语音段列表</h3>
        <p className="text-center text-secondary-color">暂无语音段</p>
      </div>
    );
  }
  
  return (
    <div className="card">
      <h3 className="text-xl font-bold mb-4">语音段列表</h3>
      <div className="overflow-auto max-h-80">
        <table className="w-full">
          <thead>
            <tr className="border-b">
              <th className="p-2 text-left">ID</th>
              <th className="p-2 text-left">开始时间</th>
              <th className="p-2 text-left">结束时间</th>
              <th className="p-2 text-left">时长</th>
              <th className="p-2 text-left">操作</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((segment) => (
              <tr 
                key={segment.segment_id} 
                className={`border-b hover:bg-gray-100 ${selectedSegment === segment.segment_id ? 'bg-blue-50' : ''}`}
              >
                <td className="p-2">{segment.segment_id}</td>
                <td className="p-2">{formatTimestamp(segment.start_timestamp_ms)}</td>
                <td className="p-2">{formatTimestamp(segment.end_timestamp_ms)}</td>
                <td className="p-2">{Math.round(segment.duration_ms)}ms</td>
                <td className="p-2">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => playSegment(segment)}
                    disabled={selectedSegment === segment.segment_id}
                  >
                    {selectedSegment === segment.segment_id ? '播放中...' : '播放'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 text-right">
        <span className="text-sm text-secondary-color">
          共 {segments.length} 个语音段
        </span>
      </div>
    </div>
  );
};

const SpeechSegmentList = memo(SpeechSegmentListComponent);

export default SpeechSegmentList;