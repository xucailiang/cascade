export class AudioStreamManager {
    constructor(wsUrl, options = {}) {
        this.wsUrl = wsUrl;
        this.options = {
            reconnectAttempts: 3,
            reconnectDelay: 2000,
            ...options
        };
        this.ws = null;
        this.reconnectCount = 0;
        this.sequenceNumber = 0;
        this.onConnected = options.onConnected || (() => {});
        this.onDisconnected = options.onDisconnected || (() => {});
        this.onMessage = options.onMessage || (() => {});
        this.onError = options.onError || (() => {});
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket连接已建立');
            this.reconnectCount = 0;
            this.onConnected();
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.onMessage(message);
            } catch (error) {
                console.error('解析WebSocket消息失败:', error);
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket连接已断开:', event.code, event.reason);
            this.onDisconnected();
            if (this.reconnectCount < this.options.reconnectAttempts) {
                this.reconnectCount++;
                console.log(`尝试重新连接... (${this.reconnectCount}/${this.options.reconnectAttempts})`);
                setTimeout(() => this.connect(), this.options.reconnectDelay);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.onError(error);
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket未连接，无法发送消息:', message);
        }
    }

    sendAudioChunk(audioData) {
        this.sendMessage({
            type: 'audio_chunk',
            data: Array.from(audioData),
            timestamp: Date.now(),
            sequence: this.sequenceNumber++
        });
    }
}