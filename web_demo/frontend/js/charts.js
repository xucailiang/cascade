export class WaveformVisualizer {
    constructor(canvasElement, options = {}) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');

        // 关键修复：确保canvas的内部绘图分辨率与其CSS显示的尺寸一致
        // 这是解决坐标偏移问题的核心
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;

        this.options = {
            waveColor: '#3498db',
            speechColor: 'rgba(243, 156, 18, 0.5)',
            backgroundColor: '#ffffff',
            timeAxisColor: '#95a5a6',
            timeAxisFont: '10px Arial',
            tooltipColor: 'rgba(0, 0, 0, 0.7)',
            tooltipFont: '12px Arial',
            ...options
        };
        this.audioBuffer = [];
        this.speechSegments = [];
        this.animationFrameId = null;
        this.startTime = 0;
        this.sampleRate = 16000;
        
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
    }
    
    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        // 确保坐标计算的准确性，将鼠标事件的屏幕坐标转换为Canvas元素的内部坐标
        const mouseX = event.clientX - rect.left;
        
        this.render(mouseX);
    }
    
    handleMouseLeave() {
        this.render();
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

    render(mouseX) {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.animationFrameId = requestAnimationFrame(() => this.draw(mouseX));
    }

    draw(mouseX) {
        const { width, height } = this.canvas;
        const halfHeight = height / 2;
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, width, height);

        // 绘制波形
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = this.options.waveColor;
        this.ctx.beginPath();
        
        const pixelsPerSample = width / this.audioBuffer.length;
        for (let i = 0; i < this.audioBuffer.length; i++) {
            const x = i * pixelsPerSample;
            const y = (this.audioBuffer[i] * halfHeight) + halfHeight;
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
            const startX = (seg.start_ms / 1000) * this.sampleRate * pixelsPerSample;
            const endX = (seg.end_ms / 1000) * this.sampleRate * pixelsPerSample;
            this.ctx.fillRect(startX, 0, endX - startX, height);
        });
        
        // 绘制时间轴和工具提示
        if (mouseX) {
            this.drawTimestampLine(mouseX);
            this.drawTooltip(mouseX);
        }
    }
    
    drawTimestampLine(mouseX) {
        const { height } = this.canvas;
        const time = (mouseX / this.canvas.width) * (this.audioBuffer.length / this.sampleRate);
        
        this.ctx.fillStyle = this.options.timeAxisColor;
        this.ctx.fillRect(mouseX, 0, 1, height);
        
        this.ctx.font = this.options.timeAxisFont;
        this.ctx.fillStyle = this.options.tooltipColor;
        this.ctx.fillText(`${time.toFixed(2)}s`, mouseX + 5, 10);
    }
    
    drawTooltip(mouseX) {
        const { width } = this.canvas;
        const timeMs = (mouseX / width) * (this.audioBuffer.length / this.sampleRate) * 1000;

        const segment = this.speechSegments.find(seg => timeMs >= seg.start_ms && timeMs <= seg.end_ms);

        if (segment) {
            const duration = (segment.end_ms - segment.start_ms) / 1000;
            const text = `语音段: ${segment.start_ms.toFixed(0)}ms - ${segment.end_ms.toFixed(0)}ms (时长: ${duration.toFixed(2)}s, 置信度: ${segment.probability.toFixed(3)})`;

            this.ctx.font = this.options.tooltipFont;
            const textWidth = this.ctx.measureText(text).width;

            let x;
            // 如果工具提示将超出画布右侧，则将其放置在光标左侧
            if (mouseX + 15 + textWidth > width) {
                x = mouseX - 15 - textWidth;
            } else {
                x = mouseX + 15;
            }
            const y = 25;

            this.ctx.fillStyle = this.options.tooltipColor;
            this.ctx.fillRect(x - 5, y - 15, textWidth + 10, 20);

            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText(text, x, y);
        }
    }

    renderFullFile(audioData ,segments) {
        this.clear();
        this.audioBuffer = audioData;
        this.speechSegments = segments;
        this.startTime = 0;
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