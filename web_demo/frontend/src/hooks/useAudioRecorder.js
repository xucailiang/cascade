import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * 音频录制钩子 (v2 - 稳定版)
 */
const useAudioRecorder = (options = {}) => {
  const {
    sampleRate = 16000,
    chunkSize = 100, // 注意：这个值在MediaRecorder中是建议值
    onDataAvailable
  } = options;
  
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const isRecordingRef = useRef(isRecording);
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null); // For ScriptProcessorNode
  
  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    
    console.log('停止录音...');
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    
    setIsRecording(false);
    setAudioLevel(0);
    console.log('录音已停止');

  }, [isRecording]);

  const startRecording = useCallback(async () => {
    if (isRecording) return;

    console.log('开始录音...');
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      
      const context = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
      audioContextRef.current = context;
      
      const source = context.createMediaStreamSource(stream);
      
      // 注意: ScriptProcessorNode 已被废弃，但对于精确的分块仍然是最可靠的方法之一
      // 在现代浏览器中，应迁移到AudioWorklet
      const bufferSize = 4096;
      const scriptNode = context.createScriptProcessor(bufferSize, 1, 1);
      
      scriptNode.onaudioprocess = (audioProcessingEvent) => {
        // 使用ref来获取最新的recording状态，避免闭包问题
        if (!isRecordingRef.current || !onDataAvailable) return;
        const inputBuffer = audioProcessingEvent.inputBuffer;
        const pcmData = inputBuffer.getChannelData(0);
        // an Int16 is two bytes
        const audioData = new Int16Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
            audioData[i] = pcmData[i] * 0x7FFF;
        }
        onDataAvailable(audioData.buffer);

        // 更新音量
        let sum = 0;
        for(let i = 0; i < pcmData.length; i++) {
           sum += pcmData[i] * pcmData[i];
        }
        const rms = Math.sqrt(sum / pcmData.length);
        setAudioLevel(rms * 2); // 乘以一个系数让变化更明显
      };
      
      source.connect(scriptNode);
      scriptNode.connect(context.destination);
      processorRef.current = scriptNode;

      setIsRecording(true);

    } catch (err) {
      console.error('开始录音失败', err);
      setError(`开始录音失败: ${err.message}`);
    }
  }, [isRecording, sampleRate, onDataAvailable]);
  
  useEffect(() => {
    // 组件卸载时确保清理资源
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    error,
    audioLevel,
    startRecording,
    stopRecording,
  };
};

export default useAudioRecorder;