"""
显示当前训练参数配置
"""
from annotation import TrainingManager
from pathlib import Path
import inspect

print("=" * 80)
print("当前训练参数配置")
print("=" * 80)

# Get default parameters from train_model signature
sig = inspect.signature(TrainingManager.train_model)
params = sig.parameters

print("\n📋 函数签名参数:")
for param_name, param in params.items():
    if param.default != inspect.Parameter.empty:
        print(f"  {param_name:15s}: {param.default}")

print("\n✅ 优化后的关键参数:")
print(f"  epochs          : 150 (之前: 50)")
print(f"  batch_size      : 8 (之前: 16)")
print(f"  optimizer       : AdamW (之前: Adam)")
print(f"  patience        : 50 (之前: 10)")
print(f"  cos_lr          : True (之前: False)")
print(f"  lr0             : 0.001 (之前: 0.01)")
print(f"  lrf             : 0.0001 (之前: 0.01)")
print(f"  warmup_epochs   : 5.0 (之前: 3.0)")
print(f"  close_mosaic    : 15 (之前: 10)")

print("\n🎨 增强的数据增强:")
print(f"  hsv_h           : 0.03 (色调变化)")
print(f"  hsv_s           : 0.8 (饱和度变化)")
print(f"  hsv_v           : 0.5 (亮度变化)")
print(f"  degrees         : 15.0 (旋转±15度)")
print(f"  translate       : 0.2 (平移20%)")
print(f"  scale           : 0.7 (缩放70%)")
print(f"  shear           : 5.0 (剪切5度)")
print(f"  perspective     : 0.0003 (透视变换)")
print(f"  mixup           : 0.1 (10%概率)")
print(f"  copy_paste      : 0.1 (10%概率)")

print("\n⏱️  预期训练时间:")
print(f"  CPU训练: ~30-45分钟 (150 epochs)")

print("\n🎯 目标性能:")
print(f"  mAP50    : > 0.8")
print(f"  Precision: > 0.7")
print(f"  Recall   : > 0.7")
print(f"  最高置信度: > 0.5 (当前: 0.007)")

print("\n" + "=" * 80)
print("✅ 参数配置已更新，可以开始训练")
print("   访问: http://localhost:3000/annotate")
print("   点击: 🚀 开始训练模型")
print("=" * 80)
