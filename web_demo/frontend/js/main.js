import { MicrophoneRecorder } from './audio.js';
import { AudioStreamManager } from './websocket.js';
import { WaveformVisualizer, PerformanceMonitor } from './charts.js';

document.addEventListener('DOMContentLoaded', () => {
    // DOM元素
    const recordBtn = document.getElementById('record-btn');
    const micSelect = document.getElementById('mic-select');
    const volumeIndicator = document.getElementById('volume-indicator');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValue = document.getElementById('threshold-value');
    const chunkDurationSelect = document.getElementById('chunk-duration-select');
    const overlapSelect = document.getElementById('overlap-select');
    const workersSelect = document.getElementById('workers-select');
    const backendSelect = document.getElementById('backend-select');
    const compensationSlider = document.getElementById('compensation-slider');
    const compensationValue = document.getElementById('compensation-value');
    const applySettingsBtn = document.getElementById('apply-settings-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const segmentsTableBody = document.getElementById('segments-table-body');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const latencyChartCanvas = document.getElementById('latency-chart');
    const throughputChartCanvas = document.getElementById('throughput-chart');
    const resourcesChartCanvas = document.getElementById('resources-chart');
    
    // 状态变量
    let isRecording = false;
    let segments = [];
    
    // 初始化模块
    const recorder = new MicrophoneRecorder({
        onAudioProcess: (audioData) => {
            if (isRecording) {
                streamManager.sendAudioChunk(audioData);
                waveformVisualizer.addAudioData(audioData);
            }
        },
        onVolumeChange: (volume) => {
            volumeIndicator.style.width = `${volume}%`;
        }
    });

    const streamManager = new AudioStreamManager(`ws://${window.location.host}/ws/ws`, {
        onConnected: () => {
            updateStatus('已连接', 'success');
            if (isRecording) {
                 streamManager.sendMessage({
                    type: 'start_recording',
                    config: getSettings()
                });
            }
        },
        onDisconnected: () => {
            updateStatus('已断开', 'danger');
        },
        onMessage: handleWebSocketMessage,
        onError: (error) => {
            console.error('WebSocket错误:', error);
            updateStatus('WebSocket错误', 'danger');
        }
    });

    const waveformVisualizer = new WaveformVisualizer(waveformCanvas);
    const performanceMonitor = new PerformanceMonitor(
        latencyChartCanvas,
        throughputChartCanvas,
        resourcesChartCanvas
    );

    // 事件监听
    recordBtn.addEventListener('click', toggleRecording);
    micSelect.addEventListener('change', () => recorder.setDevice(micSelect.value));
    applySettingsBtn.addEventListener('click', applySettings);
    thresholdSlider.addEventListener('input', () => {
        thresholdValue.textContent = thresholdSlider.value;
    });
    compensationSlider.addEventListener('input', () => {
        compensationValue.textContent = `${compensationSlider.value}ms`;
    });
    
    // 文件上传
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // 导出
    exportJsonBtn.addEventListener('click', () => exportData('json'));
    exportCsvBtn.addEventListener('click', () => exportData('csv'));
    
    // 函数
    async function toggleRecording() {
        if (isRecording) {
            // 停止录音
            recorder.stop();
            streamManager.sendMessage({ type: 'stop_recording' });
            streamManager.disconnect();
            recordBtn.innerHTML = '<i class="fas fa-play"></i> 开始录音';
            recordBtn.classList.remove('btn-danger');
            recordBtn.classList.add('btn-primary');
            isRecording = false;
            updateStatus('空闲', 'secondary');
        } else {
            // 开始录音
            await recorder.initialize();
            const success = recorder.start();
            if(success) {
                isRecording = true;
                streamManager.connect();
                recordBtn.innerHTML = '<i class="fas fa-stop"></i> 停止录音';
                recordBtn.classList.remove('btn-primary');
                recordBtn.classList.add('btn-danger');
                updateStatus('正在录音...', 'info');
                clearResults();
            } else {
                alert('无法开始录音，请检查麦克风权限。');
            }
        }
    }

    function applySettings() {
        if (isRecording) {
            streamManager.sendMessage({
                type: 'config_update',
                config: getSettings()
            });
            updateStatus('配置已更新', 'success');
        } else {
            alert('请先开始录音再应用设置');
        }
    }

    function getSettings() {
        return {
            threshold: parseFloat(thresholdSlider.value),
            chunk_duration_ms: parseInt(chunkDurationSelect.value),
            overlap_ms: parseInt(overlapSelect.value),
            workers: parseInt(workersSelect.value),
            backend: backendSelect.value,
            compensation_ms: parseInt(compensationSlider.value)
        };
    }
    
    function handleWebSocketMessage(message) {
        switch (message.type) {
            case 'vad_result':
                handleVADResult(message);
                break;
            case 'performance_metrics':
                performanceMonitor.updateMetrics(message.metrics);
                break;
            case 'status':
                if (message.status === 'recording_started') {
                    clearResults();
                }
                updateStatus(message.message, 'info');

                break;
            case 'error':
                updateStatus(message.message, 'danger');
                console.error('来自服务器的错误:', message);
                break;
        }
    }

    function handleVADResult(result) {
        if (result.is_speech) {
            waveformVisualizer.addSpeechSegment(result);
            addSegmentToTable(result);
        }
    }

    async function handleFileUpload(file) {
        updateStatus(`正在上传文件: ${file.name}`, 'info');
        
        const formData = new FormData();
        formData.append('file', file);
        
        const settings = getSettings();
        for (const key in settings) {
            formData.append(key, settings[key]);
        }

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                updateStatus('文件处理完成', 'success');
                waveformVisualizer.renderFullFile(result.audio_data, result.results);
                result.results.forEach(addSegmentToTable);
            } else {
                const error = await response.json();
                throw new Error(error.detail || '文件上传失败');
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            updateStatus(`文件上传失败: ${error.message}`, 'danger');
        }
    }

    function updateStatus(message, type) {
        statusIndicator.textContent = message;
        statusIndicator.className = `badge bg-${type}`;
    }

    function addSegmentToTable(segment) {
        segments.push(segment);
        const row = document.createElement('tr');
        const duration = (segment.end_ms - segment.start_ms) / 1000;
        
        // 延迟补偿显示信息
        let compensationInfo = '无';
        if (segment.is_compensated && segment.original_start_ms !== null) {
            const adjustedMs = segment.original_start_ms - segment.start_ms;
            compensationInfo = `${adjustedMs.toFixed(0)}ms`;
        }
        
        row.innerHTML = `
            <td>${segments.length}</td>
            <td>${(segment.start_ms / 1000).toFixed(2)}s</td>
            <td>${(segment.end_ms / 1000).toFixed(2)}s</td>
            <td>${duration.toFixed(2)}s</td>
            <td>${segment.probability.toFixed(3)}</td>
            <td>${compensationInfo}</td>
        `;
        segmentsTableBody.appendChild(row);
    }
    
    function clearResults() {
        segments = [];
        segmentsTableBody.innerHTML = '';
        waveformVisualizer.clear();
        performanceMonitor.clear();
    }
    
    function exportData(format) {
        if (segments.length === 0) {
            alert('没有数据可以导出');
            return;
        }

        if (format === 'json') {
            const jsonString = JSON.stringify(segments, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json' });
            saveAs(blob, 'vad_segments.json');
        } else if (format === 'csv') {
            let csvContent = 'data:text/csv;charset=utf-8,';
            csvContent += 'ID,StartTime(s),EndTime(s),Duration(s),Probability,DelayCompensation\n';
            segments.forEach((seg, index) => {
                const duration = (seg.end_ms - seg.start_ms) / 1000;
                let compensationInfo = '无';
                if (seg.is_compensated && seg.original_start_ms !== null) {
                    const adjustedMs = seg.original_start_ms - seg.start_ms;
                    compensationInfo = `${adjustedMs.toFixed(0)}ms`;
                }
                
                const row = [
                    index + 1,
                    (seg.start_ms / 1000).toFixed(2),
                    (seg.end_ms / 1000).toFixed(2),
                    duration.toFixed(2),
                    seg.probability.toFixed(3),
                    compensationInfo
                ].join(',');
                csvContent += row + '\n';
            });
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', 'vad_segments.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    async function init() {
        // 获取麦克风设备
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioDevices = devices.filter(device => device.kind === 'audioinput');
            audioDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `麦克风 ${micSelect.options.length + 1}`;
                micSelect.appendChild(option);
            });
        } catch (error) {
            console.error('无法获取音频设备:', error);
        }
    }

    init();
});