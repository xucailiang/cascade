export class WaveformVisualizer {
    constructor(canvasElement, options = {}) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.options = {
            waveColor: '#3498db',
            speechColor: 'rgba(243, 156, 18, 0.5)',
            backgroundColor: '#ffffff',
            timeScale: 100, // pixels per second
            ...options
        };
        this.audioBuffer = [];
        this.speechSegments = [];
        this.animationFrameId = null;
        this.startTime = 0;
    }

    addAudioData(data) {
        if (this.audioBuffer.length === 0) {
            this.startTime = Date.now();
        }
        this.audioBuffer.push(...data);
        this.render();
    }

    addSpeechSegment(segment) {
        this.speechSegments.push(segment);
        this.render();
    }

    render() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.animationFrameId = requestAnimationFrame(() => this.draw());
    }

    draw() {
        const { width, height } = this.canvas;
        const halfHeight = height / 2;
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, width, height);

        // 绘制波形
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = this.options.waveColor;
        this.ctx.beginPath();
        
        const samplesToDraw = Math.floor(width * (this.options.sampleRate || 16000) / this.options.timeScale);
        const startSample = Math.max(0, this.audioBuffer.length - samplesToDraw);
        const data = this.audioBuffer.slice(startSample);

        for (let i = 0; i < data.length; i++) {
            const x = (i / data.length) * width;
            const y = (data[i] * halfHeight) + halfHeight;
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        this.ctx.stroke();
        
        // 高亮语音段
        this.ctx.fillStyle = this.options.speechColor;
        this.speechSegments.forEach(seg => {
            const segmentStartMs = seg.start_ms - (this.startTime || 0);
            const segmentEndMs = seg.end_ms - (this.startTime || 0);
            
            const startX = (segmentStartMs / 1000) * this.options.timeScale;
            const endX = (segmentEndMs / 1000) * this.options.timeScale;
            
            const relativeStartX = startX - (startSample / (this.options.sampleRate || 16000)) * this.options.timeScale;
            const widthX = endX - startX;

            if (relativeStartX < width && relativeStartX + widthX > 0) {
                 this.ctx.fillRect(relativeStartX, 0, widthX, height);
            }
        });
    }

    renderFullFile(audioData ,segments) {
        this.clear();
        this.audioBuffer = audioData;
        this.speechSegments = segments;
        this.startTime = 0; // Для файлов начало всегда 0
        this.render();
    }

    clear() {
        this.audioBuffer = [];
        this.speechSegments = [];
        this.draw();
    }
}

export class PerformanceMonitor {
    constructor(latencyCanvas, throughputCanvas, resourcesCanvas, options = {}) {
        this.options = {
            maxDataPoints: 60,
            ...options
        };

        this.charts = {
            latency: this.createChart(latencyCanvas, '处理延迟 (ms)', ['延迟']),
            throughput: this.createChart(throughputCanvas, '吞吐量 (chunks/s)', ['吞吐量']),
            resources: this.createChart(resourcesCanvas, '资源使用', ['CPU (%)', '内存 (MB)'])
        };
    }

    createChart(canvas, title, labels) {
        return new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: labels.map((label, index) => ({
                    label: label,
                    data: [],
                    tension: 0.2,
                    borderColor: ['#3498db', '#2ecc71'][index % 2],
                    backgroundColor: ['rgba(52, 152, 219, 0.1)', 'rgba(46, 204, 113, 0.1)'][index % 2],
                    fill: true,
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    updateMetrics(metrics) {
        const now = new Date().toLocaleTimeString();

        // 更新延迟图表
        this.updateChart(this.charts.latency, now, [metrics.avg_latency_ms]);
        
        // 更新吞吐量图表
        this.updateChart(this.charts.throughput, now, [metrics.throughput_chunks_per_sec]);

        // 更新资源图表
        this.updateChart(this.charts.resources, now, [metrics.cpu_usage * 100, metrics.memory_usage_mb]);
    }

    updateChart(chart, label, data) {
        chart.data.labels.push(label);
        data.forEach((value, index) => {
            chart.data.datasets[index].data.push(value);
        });

        if (chart.data.labels.length > this.options.maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        chart.update();
    }
    
    clear() {
        for (const chartName in this.charts) {
            const chart = this.charts[chartName];
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            chart.update();
        }
    }
}