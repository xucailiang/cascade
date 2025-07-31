export class MicrophoneRecorder {
    constructor(options = {}) {
        this.config = {
            sampleRate: 16000,
            channelCount: 1,
            bufferSize: 4096,
            ...options
        };
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
        this.isRecording = false;
        this.onAudioProcess = options.onAudioProcess || (() => {});
        this.onVolumeChange = options.onVolumeChange || (() => {});
        this.deviceId = 'default';
        this.analyser = null;
        this.volumeData = null;
    }

    async initialize() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: this.deviceId,
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channelCount,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.processor = this.audioContext.createScriptProcessor(this.config.bufferSize, 1, 1);
            
            this.processor.onaudioprocess = (event) => {
                if (this.isRecording) {
                    const audioData = event.inputBuffer.getChannelData(0);
                    this.onAudioProcess(audioData);
                    this.calculateVolume(audioData);
                }
            };

            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.volumeData = new Uint8Array(this.analyser.frequencyBinCount);

            source.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            return true;
        } catch (error) {
            console.error('麦克风初始化失败:', error);
            return false;
        }
    }

    start() {
        if (!this.audioContext || !this.processor) return false;
        this.isRecording = true;
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        return true;
    }

    stop() {
        this.isRecording = false;
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
    }
    
    setDevice(deviceId) {
        this.deviceId = deviceId;
        if (this.isRecording) {
            this.stop();
            this.initialize().then(success => {
                if (success) this.start();
            });
        }
    }
    
    calculateVolume(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        const rms = Math.sqrt(sum / audioData.length);
        const volume = Math.min(100, Math.round(rms * 500)); // 放大以便显示
        this.onVolumeChange(volume);
    }
}