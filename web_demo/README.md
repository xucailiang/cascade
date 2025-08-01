# Cascade VAD Web演示界面

这是一个完整的Web界面，用于测试Cascade高性能VAD处理库的核心功能。

## 🎯 功能特性

### 实时流式处理
- **麦克风实时录音** - 浏览器WebRTC麦克风访问
- **实时VAD检测** - 毫秒级语音活动检测
- **流式结果展示** - 实时语音段可视化
- **性能监控** - 实时延迟、吞吐量指标

### 文件处理能力
- **音频文件上传** - 支持WAV、MP3等格式
- **批量处理** - 高并发文件处理演示
- **结果对比** - 不同配置参数效果对比

### 交互式配置
- **VAD参数调节** - 阈值、重叠、块大小等
- **后端切换** - ONNX/Silero后端对比
- **性能优化** - 线程数、缓冲区配置

## 📁 文件结构

```
web_demo/
├── README.md                 # 本文档
├── requirements.txt          # Python依赖
├── app.py                   # FastAPI后端服务器
├── static/
│   ├── index.html           # 主页面
│   ├── css/
│   │   ├── main.css         # 主样式
│   │   └── components.css   # 组件样式
│   └── js/
│       ├── main.js          # 主逻辑
│       ├── audio.js         # 音频处理
│       ├── websocket.js     # WebSocket通信
│       └── charts.js        # 图表可视化
├── templates/
│   └── index.html           # Jinja2模板
└── uploads/                 # 文件上传目录
```

## 🚀 快速启动

### 1. 安装依赖
```bash
# 进入项目根目录
cd /path/to/cascade

# 安装Web演示依赖
pip install fastapi uvicorn websockets python-multipart

# 或使用poetry
poetry add fastapi uvicorn websockets python-multipart
```

### 2. 启动服务器
```bash
# 启动Web服务器
python web_demo/app.py

# 或使用uvicorn
uvicorn web_demo.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 打开浏览器
```
http://localhost:8000
```

## 🔧 使用说明

### 实时麦克风测试
1. 点击"开始录音"按钮
2. 允许浏览器访问麦克风
3. 开始说话，观察实时VAD检测结果
4. 调整VAD参数观察效果变化

### 文件上传测试
1. 选择音频文件(WAV/MP3)
2. 配置处理参数
3. 上传并观察处理结果

### 性能监控
- 实时延迟图表
- 吞吐量统计
- CPU/内存使用情况
- 语音段检测准确率

## 📊 技术实现

### 前端技术栈
- **原生JavaScript** - 无框架依赖
- **WebRTC API** - 麦克风音频采集
- **WebSocket** - 实时双向通信
- **Chart.js** - 性能图表可视化
- **Bootstrap** - 响应式UI组件

### 后端技术栈
- **FastAPI** - 高性能Web框架
- **WebSocket** - 实时音频流处理
- **Cascade** - 核心VAD处理引擎
- **asyncio** - 异步并发处理

### 关键特性
- **零延迟流式处理** - WebSocket + 异步处理
- **高并发支持** - FastAPI异步架构
- **实时配置调整** - 动态参数更新
- **跨浏览器兼容** - 标准Web API

## 🎖️ 演示场景

### 场景1: 实时语音检测
```
用户说话 → 麦克风采集 → WebSocket传输 → Cascade处理 → 实时结果展示
延迟: <100ms | 准确率: >95% | 并发: 支持多用户
```

### 场景2: 文件批量处理
```
上传音频文件 → 后台队列处理 → 并行VAD分析 → 结果统计展示
支持格式: WAV/MP3 | 文件大小: <100MB | 并发处理: 4-8线程
```

### 场景3: 参数优化对比
```
调整VAD阈值 → 实时重新处理 → 结果对比展示 → 最优参数推荐
参数范围: threshold(0.1-0.9) | 重叠(16-64ms) | 块大小(256-1024ms)
```

## 💡 使用提示

1. **麦克风权限** - 首次使用需要授权浏览器麦克风访问
2. **音频格式** - 推荐使用16kHz单声道WAV格式获得最佳效果
3. **网络环境** - 实时处理需要稳定的网络连接
4. **浏览器兼容** - 推荐使用Chrome/Firefox/Safari最新版本

---

**开发状态**: 🚧 开发中 - 实现完整的实时VAD测试界面