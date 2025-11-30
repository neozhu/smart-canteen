# Smart-Canteen 开发指南

## 系统架构

### 核心理念: 视觉与业务解耦

- **AI = 眼睛**: 仅输出几何形状属性
- **UI = 大脑**: 处理所有业务逻辑
- **云端 ≠ 实时依赖**: 系统必须离线可用

### 三层架构

```
┌─────────────────────────────────────────┐
│         Frontend (Next.js)              │
│  - 操作员界面 (/)                        │
│  - 顾客屏幕 (/customer)                  │
│  - BroadcastChannel 双屏同步             │
└─────────────────────────────────────────┘
                   ↓ HTTP/REST
┌─────────────────────────────────────────┐
│       Backend (FastAPI)                 │
│  - USB 摄像头管理                        │
│  - YOLO 推理引擎                        │
│  - MJPEG 流服务                         │
│  - 订单持久化                            │
└─────────────────────────────────────────┘
                   ↓ 配置/OTA
┌─────────────────────────────────────────┐
│         Cloud Config                    │
│  - classes.json                         │
│  - price_map.json                       │
│  - model_version.json                   │
└─────────────────────────────────────────┘
```

## 开发工作流

### 1. 后端开发

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**关键端点:**
- `GET /video_feed` - MJPEG 视频流
- `GET /api/current_detection` - 最新检测结果
- `GET /api/config/price_map` - 价格映射
- `POST /api/order` - 提交订单

### 2. 前端开发

```bash
cd frontend
npm install
npm run dev
```

**开发页面:**
- http://localhost:3000 - 操作员主界面
- http://localhost:3000/customer - 顾客显示屏

**关键特性:**
- SWR 轮询检测结果 (200ms 间隔)
- N帧稳定化 (默认5帧)
- BroadcastChannel 双屏同步

### 3. 配置管理

所有配置文件位于 `backend/data/`:

**classes.json** - 形状标签定义
```json
[
  "plate_round_small",
  "plate_round_large",
  "bowl_round_small"
]
```

**price_map.json** - 业务映射
```json
{
  "plate_round_large": {
    "name": "大盘",
    "price": 15.0,
    "enabled": true
  }
}
```

**一致性规则:**
- price_map 的 key 必须存在于 classes.json
- 后端启动时自动验证
- 不匹配时记录警告

## 部署流程

### 构建生产版本

```bash
cd infra
pip install -r requirements-build.txt
python build.py
```

**输出:**
- `dist/smart-canteen/` - 完整可执行目录
- `dist/installer/install.bat` - Windows 安装脚本

### 手动部署

1. 将 `dist/smart-canteen/` 复制到目标机器
2. 确保摄像头已连接
3. 运行 `smart-canteen.exe`
4. 浏览器打开 http://localhost:8000

### 服务化部署

```powershell
# 注册 Windows 服务
sc create SmartCanteen binPath= "C:\SmartCanteen\smart-canteen.exe"
sc start SmartCanteen
```

## 调试技巧

### 查看日志

后端日志实时输出到控制台:
```
INFO - Camera 0 started
INFO - YOLO model initialized with 6 classes
INFO - Loaded price map with 6 entries
```

### 测试推理

如果没有模型文件 (`backend/models/best.onnx`),系统会以模拟模式运行,不执行真实推理。

### 顾客屏幕无响应

检查 BroadcastChannel 支持:
```javascript
if (typeof BroadcastChannel !== "undefined") {
  // 支持
}
```

## 性能优化

### 目标指标

- 推理速度: ≥ 10 FPS (目标), ≥ 5 FPS (最低)
- UI 延迟: < 500ms (目标), < 1s (最大)
- 摄像头: 720p @ 30fps

### 优化建议

1. **使用 OpenVINO** (Intel 硬件优化)
```bash
pip install openvino
```

2. **减少轮询频率** (frontend)
```typescript
refreshInterval: 500  // 从 200ms 改为 500ms
```

3. **降低视频分辨率**
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## 故障排查

### 摄像头无法打开

```python
# 尝试不同的 camera_id
camera = CameraManager(camera_id=1)  # 或 2, 3...
```

### 前端无法连接后端

检查 `.env.local`:
```
NEXT_PUBLIC_API_BASE=http://192.168.1.100:8000
```

### 双屏不同步

确保两个浏览器窗口来自同一域名(不能跨域使用 BroadcastChannel)。

## 贡献指南

1. 所有新功能需要书面 spec
2. 遵循宪章中的架构约束
3. 保持视觉与业务逻辑分离
4. 测试离线模式

## 资源

- 项目宪章: `.specify/memory/constitution.md`
- API 文档: http://localhost:8000/docs (FastAPI 自动生成)
