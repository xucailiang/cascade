# Cascade VAD WebSocket演示应用

这是一个基于WebSocket的Cascade VAD（语音活动检测）演示应用，用于展示Cascade对多客户端WebSocket连接的支持能力。

## 系统要求

- Python 3.8+
- Node.js 16+
- pnpm (推荐) 或 npm

## 项目结构

```
web_demo/
├── server.py         # FastAPI WebSocket服务器
├── frontend/         # React前端应用
    ├── src/
    │   ├── hooks/    # React钩子
    │   ├── components/ # React组件
    │   └── App.jsx   # 主应用组件
    ├── index.html    # HTML入口
    └── vite.config.js # Vite配置
```

## 安装步骤

### 后端

确保已安装Cascade库：

```bash
pip install cascade-vad
```

### 前端

在frontend目录下安装依赖：

```bash
cd web_demo/frontend
pnpm install  # 或 npm install
```

## 运行说明

### 安装额外依赖并启动后端服务

```bash

poetry install --extras "demo"

cd web_demo
python server.py
```

服务器将在 `ws://localhost:8000/ws` 启动WebSocket端点。

### 启动前端应用

```bash
cd web_demo/frontend
pnpm dev  # 或 npm run dev
```

前端应用将在 `http://localhost:3000` 启动。

## 功能说明

1. **实时音频处理**：通过浏览器麦克风捕获音频并发送到服务器进行VAD处理
2. **VAD结果可视化**：实时显示VAD检测结果
3. **语音段管理**：显示检测到的语音段并支持回放
4. **VAD配置调整**：支持动态调整VAD参数

## VAD配置选项

- **阈值**：VAD检测的灵敏度阈值
- **窗口大小**：VAD分析的时间窗口大小
- **最小语音段长度**：被识别为有效语音的最小持续时间
- **静音填充**：语音段前后的静音填充时间

## 多客户端支持

本演示应用支持多个客户端同时连接，每个客户端将获得独立的Cascade处理实例，互不干扰。