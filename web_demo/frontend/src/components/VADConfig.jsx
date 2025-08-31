import React, { useState, memo } from 'react';

/**
 * VAD配置组件
 * 
 * @param {Object} props - 组件属性
 * @param {Object} props.defaultConfig - 默认配置
 * @param {Function} props.onConfigChange - 配置变更回调
 * @returns {JSX.Element} VAD配置组件
 */
const VADConfigComponent = ({
  defaultConfig = {
    vad_threshold: 0.5,
    speech_pad_ms: 100,
    min_silence_duration_ms: 100
  },
  onConfigChange
}) => {
  const [config, setConfig] = useState(defaultConfig);
  const [isExpanded, setIsExpanded] = useState(false);
  
  // 处理配置变更
  const handleConfigChange = (e) => {
    const { name, value, type } = e.target;
    
    // 根据输入类型转换值
    const newValue = type === 'number' || type === 'range' 
      ? parseFloat(value) 
      : value;
    
    // 更新配置
    const newConfig = {
      ...config,
      [name]: newValue
    };
    
    setConfig(newConfig);
    
    // 调用回调
    if (onConfigChange) {
      onConfigChange(newConfig);
    }
  };
  
  // 重置配置
  const resetConfig = () => {
    setConfig(defaultConfig);
    
    // 调用回调
    if (onConfigChange) {
      onConfigChange(defaultConfig);
    }
  };
  
  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">VAD配置</h3>
        <button 
          className="btn btn-primary"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '收起' : '展开'}
        </button>
      </div>
      
      {isExpanded && (
        <div className="config-form">
          <div className="form-group">
            <label htmlFor="vad_threshold" className="mb-2 block">
              VAD检测阈值: {config.vad_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              id="vad_threshold"
              name="vad_threshold"
              min="0"
              max="1"
              step="0.01"
              value={config.vad_threshold}
              onChange={handleConfigChange}
              className="form-control"
            />
            <div className="flex justify-between text-sm">
              <span>0.0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
            <p className="text-sm text-secondary-color mt-1">
              较高的阈值会减少误检测，但可能会漏掉轻声说话
            </p>
          </div>
          
          <div className="form-group mt-4">
            <label htmlFor="speech_pad_ms" className="mb-2 block">
              语音段填充时长(ms): {config.speech_pad_ms}
            </label>
            <input
              type="number"
              id="speech_pad_ms"
              name="speech_pad_ms"
              min="0"
              max="1000"
              step="10"
              value={config.speech_pad_ms}
              onChange={handleConfigChange}
              className="form-control"
            />
            <p className="text-sm text-secondary-color mt-1">
              语音段两侧填充的时长，增大此值可获得更完整的语音
            </p>
          </div>
          
          <div className="form-group mt-4">
            <label htmlFor="min_silence_duration_ms" className="mb-2 block">
              最小静音时长(ms): {config.min_silence_duration_ms}
            </label>
            <input
              type="number"
              id="min_silence_duration_ms"
              name="min_silence_duration_ms"
              min="0"
              max="1000"
              step="10"
              value={config.min_silence_duration_ms}
              onChange={handleConfigChange}
              className="form-control"
            />
            <p className="text-sm text-secondary-color mt-1">
              短于此时长的静音将被视为语音的一部分
            </p>
          </div>
          
          <div className="mt-4 text-right">
            <button 
              className="btn btn-danger"
              onClick={resetConfig}
            >
              重置配置
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const VADConfig = memo(VADConfigComponent);

export default VADConfig;