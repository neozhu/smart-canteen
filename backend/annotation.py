"""
Annotation and Training Module for Smart-Canteen
Handles data collection, annotation, and model training
"""
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class AnnotationManager:
    """Manages dataset annotation and storage"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_dir = data_dir / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
    def capture_sample(self, frame: np.ndarray, label: str, class_index: int, bbox: Optional[List[float]] = None) -> str:
        """
        Capture and save an annotated sample
        
        Args:
            frame: Camera frame
            label: Class label (e.g., 'plate_round_large')
            class_index: Index in classes.json
            bbox: Optional bounding box [x1, y1, x2, y2] in pixels. If None, auto-detect or use full frame
            
        Returns:
            Filename of saved image
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{label}_{timestamp}_{unique_id}"
        
        # Save image
        image_path = self.images_dir / f"{filename}.jpg"
        cv2.imwrite(str(image_path), frame)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create YOLO format label
        # YOLO format: class_id center_x center_y width height (normalized 0-1)
        
        if bbox is not None and len(bbox) == 4:
            # Use provided bbox [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            # Convert to center format and normalize
            center_x = ((x1 + x2) / 2) / w
            center_y = ((y1 + y2) / 2) / h
            bbox_width = (x2 - x1) / w
            bbox_height = (y2 - y1) / h
            
            # Clamp to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            bbox_width = max(0.01, min(1, bbox_width))
            bbox_height = max(0.01, min(1, bbox_height))
            
            logger.info(f"Using provided bbox: center=({center_x:.3f},{center_y:.3f}), size=({bbox_width:.3f},{bbox_height:.3f})")
        else:
            # Try to auto-detect object using simple background subtraction
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Use Otsu's thresholding
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(largest_contour)
                    
                    # Check if detected object is reasonable size (not too small/large)
                    area_ratio = (bw * bh) / (w * h)
                    if 0.05 < area_ratio < 0.95:
                        # Use detected bbox
                        center_x = (x + bw / 2) / w
                        center_y = (y + bh / 2) / h
                        bbox_width = bw / w
                        bbox_height = bh / h
                        logger.info(f"Auto-detected bbox: center=({center_x:.3f},{center_y:.3f}), size=({bbox_width:.3f},{bbox_height:.3f})")
                    else:
                        raise ValueError("Detected object size unreasonable")
                else:
                    raise ValueError("No contours found")
                    
            except Exception as e:
                # Fallback to conservative full-frame bbox
                logger.warning(f"Auto-detection failed: {e}, using 70% center bbox")
                center_x = 0.5
                center_y = 0.5
                bbox_width = 0.7
                bbox_height = 0.7
        
        label_content = f"{class_index} {center_x} {center_y} {bbox_width} {bbox_height}\n"
        
        label_path = self.labels_dir / f"{filename}.txt"
        with open(label_path, 'w') as f:
            f.write(label_content)
        
        logger.info(f"Saved annotation: {filename} with label {label}")
        
        return filename
    
    def get_stats(self, classes: List[str]) -> Dict:
        """Get annotation statistics for all classes"""
        stats = {}
        
        for cls in classes:
            # Find all images for this class
            samples = list(self.images_dir.glob(f"{cls}_*.jpg"))
            stats[cls] = {
                "name": cls,
                "count": len(samples),
                "samples": [s.name for s in samples]
            }
        
        return stats
    
    def delete_sample(self, filename: str) -> bool:
        """Delete a sample (image + label)"""
        image_path = self.images_dir / filename
        label_path = self.labels_dir / f"{filename.replace('.jpg', '.txt')}"
        
        try:
            if image_path.exists():
                image_path.unlink()
            if label_path.exists():
                label_path.unlink()
            logger.info(f"Deleted sample: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False
    
    def prepare_dataset(self, classes: List[str], train_ratio: float = 0.8):
        """
        Prepare dataset for training (train/val split)
        
        Args:
            classes: List of class names
            train_ratio: Ratio of training data (default 0.8)
        """
        from sklearn.model_selection import train_test_split
        
        # Create train/val directories
        train_images = self.dataset_dir / "train" / "images"
        train_labels = self.dataset_dir / "train" / "labels"
        val_images = self.dataset_dir / "val" / "images"
        val_labels = self.dataset_dir / "val" / "labels"
        
        for dir_path in [train_images, train_labels, val_images, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split each class
        for cls in classes:
            samples = list(self.images_dir.glob(f"{cls}_*.jpg"))
            
            if len(samples) < 2:
                continue
            
            # Split
            train_samples, val_samples = train_test_split(
                samples,
                train_size=train_ratio,
                random_state=42
            )
            
            # Copy to train
            for sample in train_samples:
                shutil.copy(sample, train_images / sample.name)
                label_file = sample.with_suffix('.txt').name
                shutil.copy(
                    self.labels_dir / label_file,
                    train_labels / label_file
                )
            
            # Copy to val
            for sample in val_samples:
                shutil.copy(sample, val_images / sample.name)
                label_file = sample.with_suffix('.txt').name
                shutil.copy(
                    self.labels_dir / label_file,
                    val_labels / label_file
                )
        
        logger.info(f"Dataset prepared: {len(list(train_images.glob('*.jpg')))} train, "
                   f"{len(list(val_images.glob('*.jpg')))} val")


class TrainingManager:
    """Manages model training process"""
    
    def __init__(self, data_dir: Path, models_dir: Path):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.dataset_dir = data_dir / "dataset"
        self.training_dir = data_dir / "training"
        self.training_dir.mkdir(exist_ok=True)
        
        self.current_task = None
        self.progress = 0
        self.status = "idle"
        self.message = ""
    
    def create_dataset_yaml(self, classes: List[str]) -> Path:
        """Create dataset.yaml for YOLO training"""
        yaml_content = {
            "path": str(self.dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes),
            "names": classes
        }
        
        yaml_path = self.dataset_dir / "dataset.yaml"
        
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        logger.info(f"Created dataset.yaml with {len(classes)} classes")
        return yaml_path
    
    def train_model(
        self,
        classes: List[str],
        epochs: int = 150,
        img_size: int = 640,
        batch_size: int = 8
    ) -> str:
        """
        Train YOLO model
        
        Args:
            classes: List of class names
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Training batch size
            
        Returns:
            Task ID for progress tracking
        """
        task_id = str(uuid.uuid4())
        self.current_task = task_id
        self.status = "preparing"
        self.progress = 0
        self.message = "准备数据集..."
        
        try:
            # Prepare dataset
            annotation_manager = AnnotationManager(self.data_dir)
            annotation_manager.prepare_dataset(classes)
            
            # Create dataset.yaml
            yaml_path = self.create_dataset_yaml(classes)
            
            self.progress = 10
            self.message = "初始化模型..."
            
            # Load base model (YOLOv8n)
            model = YOLO('yolov8n.pt')
            
            self.status = "training"
            self.progress = 15
            self.message = "开始训练..."
            
            # Train with optimized parameters
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device='cpu',  # Use 'cuda' if GPU available
                project=str(self.training_dir),
                name=f"run_{task_id[:8]}",
                patience=50,  # Increased from 10 for better convergence
                save=True,
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',  # Changed to AdamW for better generalization
                verbose=True,
                seed=42,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,  # Enable cosine learning rate for smoother training
                close_mosaic=15,  # Increased for more normal training at end
                resume=False,
                amp=False,
                fraction=1.0,
                profile=False,
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.0,
                val=True,
                split='val',
                save_json=False,
                save_hybrid=False,
                conf=None,
                iou=0.7,
                max_det=300,
                half=False,
                dnn=False,
                plots=True,
                source=None,
                show=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                vid_stride=1,
                line_width=None,
                visualize=False,
                augment=False,
                agnostic_nms=False,
                classes=None,
                retina_masks=False,
                boxes=True,
                format='torchscript',
                keras=False,
                optimize=False,
                int8=False,
                dynamic=False,
                simplify=False,
                opset=None,
                workspace=4,
                nms=False,
                lr0=0.001,  # Reduced initial learning rate for stability
                lrf=0.0001,  # Lower final learning rate
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5.0,  # Increased warmup for stable start
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                pose=12.0,
                kobj=1.0,
                label_smoothing=0.0,
                nbs=64,
                # Enhanced data augmentation for better generalization
                hsv_h=0.03,  # Increased hue variation
                hsv_s=0.8,   # Increased saturation variation
                hsv_v=0.5,   # Increased brightness variation
                degrees=15.0,  # Add rotation augmentation
                translate=0.2,  # Increased translation
                scale=0.7,  # Increased scale variation
                shear=5.0,  # Add shear augmentation
                perspective=0.0003,  # Add perspective transformation
                flipud=0.0,  # No vertical flip (food items)
                fliplr=0.5,  # Keep horizontal flip
                mosaic=1.0,  # Keep mosaic augmentation
                mixup=0.1,  # Add mixup for better generalization
                copy_paste=0.1,  # Add copy-paste augmentation
                cfg=None,
                tracker='botsort.yaml',
            )
            
            self.progress = 90
            self.message = "导出模型..."
            
            # Export to ONNX
            best_model_path = self.training_dir / f"run_{task_id[:8]}" / "weights" / "best.pt"
            
            if not best_model_path.exists():
                raise RuntimeError("训练完成但未找到模型文件")
            
            # Load trained model
            trained_model = YOLO(str(best_model_path))
            
            # Export to ONNX
            onnx_path = trained_model.export(format='onnx')
            
            # Copy to models directory
            final_model_path = self.models_dir / "best.onnx"
            shutil.copy(onnx_path, final_model_path)
            
            self.progress = 100
            self.status = "completed"
            self.message = "训练完成！"
            
            logger.info(f"Training completed: {task_id}")
            
            return task_id
            
        except Exception as e:
            self.status = "failed"
            self.message = f"训练失败: {str(e)}"
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def get_progress(self, task_id: str) -> Dict:
        """Get training progress"""
        if task_id != self.current_task:
            return {
                "progress": 0,
                "status": "not_found",
                "message": "Task not found"
            }
        
        return {
            "progress": self.progress,
            "status": self.status,
            "message": self.message
        }
