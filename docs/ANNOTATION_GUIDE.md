# 数据标注与模型训练使用指南

## 功能概述

Smart-Canteen 现在支持自主数据标注和模型训练，无需手动编写标注文件。系统会自动处理从数据采集到模型部署的完整流程。

## 工作流程

```
📸 数据采集 → 🏷️ 自动标注 → 🚀 模型训练 → 📦 模型部署 → ✅ 推理使用
```

---

## 使用步骤

### 1. 进入标注页面

从主页点击右上角 **"📝 数据标注"** 按钮，进入标注界面。

页面包含：
- **左侧**: 实时摄像头预览 + 拍照控制
- **右侧**: 标注统计 + 训练控制

---

### 2. 数据采集

#### 操作流程

1. **准备物品**: 将需要标注的物品（盘子、碗、托盘等）放在摄像头前
2. **选择标签**: 在下拉菜单中选择对应的标签类型（如 `plate_round_large`）
3. **拍照保存**: 点击 **"📸 拍照标注"** 按钮
4. **重复采集**: 对同一物品从不同角度拍摄多张照片

#### 采集建议

- ✅ **每个类别至少 20 张照片**（建议 30-50 张）
- ✅ **多角度拍摄**: 正面、侧面、倾斜等
- ✅ **不同光照**: 明亮、阴影、逆光
- ✅ **不同位置**: 画面中央、边缘、靠近、远离
- ✅ **真实场景**: 模拟实际使用时的摆放方式

#### 自动标注原理

系统会自动：
- 保存当前帧为 JPG 图像
- 生成 YOLO 格式的标注文件（`.txt`）
- 使用整个画面作为检测区域（90% 覆盖）
- 按类别索引记录标签信息

**文件存储位置**:
```
backend/data/dataset/
├── images/          # 原始图像
│   ├── plate_round_large_20241130_143022_a1b2c3d4.jpg
│   └── bowl_round_small_20241130_143045_e5f6g7h8.jpg
└── labels/          # YOLO 标注
    ├── plate_round_large_20241130_143022_a1b2c3d4.txt
    └── bowl_round_small_20241130_143045_e5f6g7h8.txt
```

---

### 3. 查看统计

右侧面板实时显示：

- **总样本数**: 所有类别的图像总数
- **每个类别**:
  - 当前样本数
  - 完成度进度条（目标 20 张）
  - 最近 5 个样本（可点击删除）

#### 删除样本

如果拍摄失败或质量不佳：
- 点击样本文件名后的 **×** 按钮删除
- 系统会同时删除图像和标注文件

---

### 4. 开始训练

当采集足够样本后（建议总数 ≥ 100 张）：

1. 点击 **"🚀 开始训练模型"** 按钮
2. 确认训练参数
3. 等待训练完成（进度条实时更新）

#### 训练过程

系统会自动执行：

1. **数据准备** (10%)
   - 按 80:20 比例分割训练集和验证集
   - 复制图像和标注到 `train/` 和 `val/` 目录
   
2. **初始化模型** (15%)
   - 加载预训练的 YOLOv8n 基础模型
   - 创建 `dataset.yaml` 配置文件

3. **训练循环** (15-90%)
   - 默认训练 50 个 epoch
   - 自动保存最佳模型 (`best.pt`)
   - 每个 epoch 更新进度

4. **模型导出** (90-100%)
   - 将 PyTorch 模型导出为 ONNX 格式
   - 复制到 `backend/models/best.onnx`

#### 训练参数

默认配置（可在 `annotation.py` 中修改）：

```python
epochs = 50           # 训练轮数
img_size = 640        # 输入图像尺寸
batch_size = 8        # 批次大小（可根据内存调整）
optimizer = 'Adam'    # 优化器
```

#### 训练时间估算

| 样本数 | 硬件配置 | 预计时间 |
|--------|----------|----------|
| 100-200 | CPU | 10-20 分钟 |
| 200-500 | CPU | 20-40 分钟 |
| 100-200 | GPU | 2-5 分钟 |
| 200-500 | GPU | 5-10 分钟 |

---

### 5. 模型部署

训练完成后：

1. **自动部署**: 新模型已保存到 `backend/models/best.onnx`
2. **重启后端**: 
   ```bash
   # 停止当前运行的 main.py (Ctrl+C)
   # 重新启动
   python main.py
   ```
3. **验证**: 返回主页，系统会自动加载新模型进行推理

---

## 文件结构

```
backend/data/
├── dataset/                    # 数据集根目录
│   ├── images/                 # 原始标注数据
│   │   ├── plate_*.jpg
│   │   └── bowl_*.jpg
│   ├── labels/                 # YOLO 标注文件
│   │   ├── plate_*.txt
│   │   └── bowl_*.txt
│   ├── train/                  # 训练集（训练时自动生成）
│   │   ├── images/
│   │   └── labels/
│   ├── val/                    # 验证集（训练时自动生成）
│   │   ├── images/
│   │   └── labels/
│   └── dataset.yaml            # YOLO 配置文件
├── training/                   # 训练输出
│   └── run_xxxxxx/
│       ├── weights/
│       │   ├── best.pt         # 最佳模型
│       │   └── last.pt         # 最后模型
│       └── results.png         # 训练曲线
└── classes.json                # 类别定义（不可修改）
```

---

## 标注格式说明

### YOLO 格式

每个图像对应一个 `.txt` 文件，格式为：

```
<class_id> <center_x> <center_y> <width> <height>
```

**示例**:
```
0 0.5 0.5 0.9 0.9
```

参数说明：
- `class_id`: 类别索引（对应 `classes.json` 中的位置）
- `center_x`: 边界框中心 X 坐标（归一化，0-1）
- `center_y`: 边界框中心 Y 坐标（归一化，0-1）
- `width`: 边界框宽度（归一化，0-1）
- `height`: 边界框高度（归一化，0-1）

**为什么使用全画面标注？**

因为采集时画面中只有一个目标物品，所以：
- 简化标注流程（无需手动绘制边界框）
- 提高标注效率（一键完成）
- 适合单物品检测场景

---

## 高级操作

### 修改类别

如果需要添加/删除/重命名类别：

1. 编辑 `backend/data/classes.json`
2. 同步更新 `backend/data/price_map.json`
3. **重新标注数据**（旧数据不兼容）
4. 重新训练模型

### 调整训练参数

编辑 `backend/annotation.py` 的 `train_model()` 方法：

```python
def train_model(
    self,
    classes: List[str],
    epochs: int = 100,        # 增加 epoch
    img_size: int = 640,
    batch_size: int = 16      # GPU 可增大 batch
):
```

### 使用 GPU 加速

修改 `annotation.py` 第 205 行：

```python
device='cuda',  # 改为 'cuda'
```

确保已安装 CUDA 版本的 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 查看训练日志

训练输出保存在：
```
backend/data/training/run_xxxxxxxx/
├── results.png        # 训练曲线图
├── confusion_matrix.png  # 混淆矩阵
└── weights/
    ├── best.pt        # 最佳权重
    └── last.pt        # 最后权重
```

---

## 故障排查

### 问题 1: 拍照无响应

**原因**: 摄像头未启动或帧读取失败

**解决**:
```bash
# 检查后端日志
# 应该看到: "Camera 0 started"
```

### 问题 2: 训练失败 - 内存不足

**原因**: batch_size 过大

**解决**:
```python
# 减小 batch_size
batch_size = 4  # 或更小
```

### 问题 3: 训练完成但推理无效果

**原因**: 后端未重载模型

**解决**:
```bash
# 重启后端服务
python main.py
```

### 问题 4: 类别不匹配警告

**原因**: `classes.json` 与已标注数据不一致

**解决**:
- 删除旧数据: `backend/data/dataset/`
- 重新标注

---

## 最佳实践

### 数据质量

✅ **好的样本**:
- 物品居中，清晰可见
- 光照均匀
- 背景简洁
- 无遮挡

❌ **差的样本**:
- 模糊、失焦
- 过曝或过暗
- 物品被遮挡
- 多个物品混在一起

### 训练策略

1. **初期快速迭代**:
   - 每类 10 张 → 快速训练 → 测试效果
   - 识别问题类别

2. **针对性增强**:
   - 对混淆类别增加样本
   - 多角度、多场景采集

3. **最终优化**:
   - 每类 30-50 张
   - 完整训练 100 epoch

---

## API 参考

### 拍照标注
```http
POST /api/annotation/capture
Content-Type: application/json

{
  "label": "plate_round_large"
}
```

### 获取统计
```http
GET /api/annotation/stats
```

### 删除样本
```http
DELETE /api/annotation/sample
Content-Type: application/json

{
  "label": "plate_round_large",
  "filename": "plate_round_large_20241130_143022_a1b2c3d4.jpg"
}
```

### 开始训练
```http
POST /api/training/start
```

### 查询进度
```http
GET /api/training/progress/{task_id}
```

---

## 总结

通过这套标注+训练系统，你可以：

✅ **无需专业工具** - 浏览器内完成所有操作  
✅ **自动化流程** - 从采集到部署全自动  
✅ **实时反馈** - 进度条、统计实时更新  
✅ **灵活扩展** - 随时添加新类别  
✅ **生产就绪** - 训练后直接用于推理  

任何问题请参考后端日志或查看 API 文档: http://localhost:8000/docs
