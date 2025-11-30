"""
验证标注数据的正确性
"""
import json
import cv2
import numpy as np
from pathlib import Path
import random

def verify_annotations():
    """可视化验证标注数据"""
    
    # Paths
    base_dir = Path(__file__).parent
    classes_path = base_dir / "data" / "classes.json"
    images_dir = base_dir / "data" / "dataset" / "images"
    labels_dir = base_dir / "data" / "dataset" / "labels"
    
    # Load classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    print("=" * 80)
    print("标注数据验证")
    print("=" * 80)
    
    # Get all images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"\n总共找到 {len(image_files)} 张图片")
    
    # Statistics
    class_counts = {cls: 0 for cls in classes}
    
    # Check random samples
    samples = random.sample(image_files, min(10, len(image_files)))
    
    print(f"\n随机检查 {len(samples)} 个样本:\n")
    
    for img_path in samples:
        # Get corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            print(f"❌ {img_path.name}: 缺少标注文件!")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ {img_path.name}: 无法读取图片!")
            continue
        
        h, w = img.shape[:2]
        
        # Load label
        with open(label_path, 'r') as f:
            label_line = f.read().strip()
        
        if not label_line:
            print(f"⚠️  {img_path.name}: 标注文件为空!")
            continue
        
        parts = label_line.split()
        if len(parts) != 5:
            print(f"❌ {img_path.name}: 标注格式错误 (应该是5个值)")
            continue
        
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        
        # Validate
        if class_id < 0 or class_id >= len(classes):
            print(f"❌ {img_path.name}: 类别ID {class_id} 超出范围!")
            continue
        
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
            print(f"❌ {img_path.name}: 坐标超出范围 [0,1]!")
            continue
        
        class_name = classes[class_id]
        class_counts[class_name] += 1
        
        # Convert to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = w * h
        bbox_ratio = bbox_area / image_area
        
        # Check if bbox is reasonable
        if bbox_ratio < 0.01 or bbox_ratio > 0.95:
            status = "⚠️"
        else:
            status = "✅"
        
        print(f"{status} {img_path.name}")
        print(f"   类别: [{class_id}] {class_name}")
        print(f"   归一化坐标: center=({cx:.3f}, {cy:.3f}), size=({bw:.3f}, {bh:.3f})")
        print(f"   像素坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"   Bbox大小: {x2-x1}x{y2-y1} ({bbox_ratio*100:.1f}% of image)")
        print()
    
    # Count all labels
    print("\n" + "=" * 80)
    print("完整数据集统计:")
    print("=" * 80)
    
    all_class_counts = {cls: 0 for cls in classes}
    missing_labels = 0
    invalid_labels = 0
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            missing_labels += 1
            continue
        
        try:
            with open(label_path, 'r') as f:
                label_line = f.read().strip()
            
            if label_line:
                class_id = int(label_line.split()[0])
                if 0 <= class_id < len(classes):
                    all_class_counts[classes[class_id]] += 1
                else:
                    invalid_labels += 1
        except:
            invalid_labels += 1
    
    print(f"\n总图片数: {len(image_files)}")
    print(f"缺少标注: {missing_labels}")
    print(f"无效标注: {invalid_labels}")
    print(f"\n各类别样本数:")
    for cls, count in all_class_counts.items():
        bar = "█" * (count // 2)
        print(f"  {cls:20s}: {count:3d} {bar}")
    
    print("\n" + "=" * 80)
    
    # Visualize one sample per class
    print("\n生成可视化验证图片...")
    output_dir = base_dir / "verification_output"
    output_dir.mkdir(exist_ok=True)
    
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for class_id, class_name in enumerate(classes):
        # Find images of this class
        class_images = []
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_line = f.read().strip()
                if label_line and int(label_line.split()[0]) == class_id:
                    class_images.append(img_path)
        
        if not class_images:
            continue
        
        # Pick random sample
        sample_path = random.choice(class_images)
        label_path = labels_dir / (sample_path.stem + ".txt")
        
        # Load and draw
        img = cv2.imread(str(sample_path))
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            label_line = f.read().strip()
        
        parts = label_line.split()
        cx, cy, bw, bh = map(float, parts[1:])
        
        # Convert to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Draw bbox
        color = colors[class_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw center
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        cv2.circle(img, (cx_px, cy_px), 5, color, -1)
        
        # Add label text
        label_text = f"{class_name}"
        cv2.putText(img, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Save
        output_path = output_dir / f"{class_name}_verification.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"  ✅ {class_name}: {output_path.name}")
    
    print(f"\n可视化图片已保存到: {output_dir}")
    print("\n建议:")
    print("  1. 检查 verification_output 文件夹中的图片")
    print("  2. 确认标注框是否准确覆盖物体")
    print("  3. 如果标注框太小或太大,需要重新标注")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    verify_annotations()
