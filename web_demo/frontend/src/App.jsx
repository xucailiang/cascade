import React, { useState, useCallback, useMemo } from 'react';
import VADConfig from './components/VADConfig';
import SpeechSegmentList from './components/SpeechSegmentList';
import useWebSocket from './hooks/useWebSocket';
import useAudioRecorder from './hooks/useAudioRecorder';

/**
 * Cascade VAD演示应用 (v3 - 终极简化版)
 */
const App = () => {
  // --- 状态管理 ---
  const [clientId, setClientId] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('未连接');
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [vadConfig, setVadConfig] = useState({
    vad_threshold: 0.5,
    speech_pad_ms: 100,
    min_silence_duration_ms: 100,
    sample_rate: 16000
  });

  const [segments, setSegments] = useState([]);
  const [error, setError] = useState(null);

  // --- WebSocket 通信 ---
  const handleWsMessage = useCallback((data) => {
    switch (data.type || data.result_type) {
      case 'connection_ready':
        setConnectionStatus('已连接');
        setClientId(data.client_id);
        setError(null);
        break;
      case 'segment':
        if(data.segment) {
            setSegments(prev => {
                const existing = prev.find(s => s.segment_id === data.segment.segment_id);
                if (existing) {
                    return prev.map(s => s.segment_id === data.segment.segment_id ? data.segment : s);
                }
                return [...prev, data.segment];
            });
        }
        break;
      case 'error':
        setError(data.message);
        break;
      default:
        break;
    }
  }, []);

  const handleWsClose = useCallback(() => {
    setConnectionStatus('已断开');
    setIsSessionActive(false);
    setClientId(null);
  }, []);

  const {
    isConnected,
    error: wsError,
    connect,
    disconnect,
    sendMessage,
    sendBinary,
  } = useWebSocket('ws://localhost:8000/ws/new', {
    onMessage: handleWsMessage,
    onClose: handleWsClose,
    onError: () => setConnectionStatus('连接错误'),
  });

  // --- 音频录制 ---
  const { isRecording, startRecording, stopRecording } = useAudioRecorder({
    sampleRate: vadConfig.sample_rate,
    onDataAvailable: sendBinary // 直接将音频数据发送函数传给钩子
  });
  
  // --- UI 事件处理 ---
  const handleToggleConnection = useCallback(() => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  }, [isConnected, connect, disconnect]);

  const handleToggleRecording = useCallback(() => {
    if (isSessionActive) {
      sendMessage({ type: 'stop' });
      stopRecording();
      setIsSessionActive(false);
    } else {
      if (!isConnected) {
        setError("请先连接服务器");
        return;
      }
      setSegments([]);
      sendMessage({ type: 'start', config: vadConfig });
      startRecording();
      setIsSessionActive(true);
    }
  }, [isConnected, isSessionActive, sendMessage, startRecording, stopRecording, vadConfig]);

  const handleConfigChange = useCallback((newConfig) => {
    setVadConfig(newConfig);
    // 当会话激活时，立即发送新配置
    if (isSessionActive) {
      sendMessage({ type: 'start', config: newConfig });
    }
  }, [isSessionActive, sendMessage, vadConfig]); // vadConfig必须加入依赖项
  
  const clearResults = useCallback(() => {
    setSegments([]);
  }, []);

  const displayError = error || wsError;

  return (
    <div className="container mx-auto py-4">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-center">Cascade VAD演示 (v3)</h1>
        <p className="text-center text-secondary-color">语音活动检测(VAD)实时演示</p>
      </header>

      <div className="card mb-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-xl font-bold">连接状态</h3>
            <p className="mt-2">
              状态: <span className={`font-bold ${isConnected ? 'text-success-color' : 'text-danger-color'}`}>
                {connectionStatus}
              </span>
              {clientId && <span className="ml-2">客户端ID: {clientId}</span>}
            </p>
          </div>
          <div className="flex gap-2">
            <button className={`btn ${isConnected ? 'btn-danger' : 'btn-primary'}`} onClick={handleToggleConnection}>
              {isConnected ? '断开连接' : '连接服务器'}
            </button>
            <button className={`btn ${isSessionActive ? 'btn-danger' : 'btn-primary'}`} onClick={handleToggleRecording} disabled={!isConnected}>
              {isSessionActive ? '停止录音' : '开始录音'}
            </button>
            <button className="btn btn-secondary" onClick={clearResults}>清除结果</button>
          </div>
        </div>
        {displayError && <div className="alert alert-danger mt-4"><strong>错误:</strong> {String(displayError)}</div>}
      </div>

      <VADConfig defaultConfig={vadConfig} onConfigChange={handleConfigChange} />
      
      <SpeechSegmentList segments={segments} sampleRate={vadConfig.sample_rate} />
      
      <footer className="mt-8 text-center text-sm text-secondary-color">
        <p>Cascade VAD演示 &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
};

export default App;