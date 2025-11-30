"""
Smart-Canteen Backend Service
FastAPI service for camera management, YOLO inference, and order handling
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from annotation import AnnotationManager, TrainingManager, get_center_square_bbox

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR.parent / "frontend" / "out"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart-Canteen Backend")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
camera = None
model = None
latest_detection = []
classes = []
price_map = {}
annotation_manager = None
training_manager = None
is_training = False  # Flag to pause inference during training
assist_model = None  # YOLOv8n pretrained model for annotation assistance
enable_inference_debug = True  # Control inference debug logging


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class Order(BaseModel):
    items: List[dict]
    total: float
    timestamp: str


class CameraManager:
    """Manages USB camera capture"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.running = False
        
    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera to 720p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.running = True
        logger.info(f"Camera {self.camera_id} started")
        
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")
        
    def read_frame(self):
        """Read a frame from camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            return frame
        return None


class YOLOInference:
    """YOLO model inference wrapper"""
    
    def __init__(self, model_path: str, classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = None
        self.input_size = 640
        
        # Load ONNX model
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded: {model_path}")
            logger.info(f"Input: {self.input_name}, Outputs: {self.output_names}")
            logger.info(f"Classes ({len(classes)}): {classes}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference"""
        global enable_inference_debug
        if enable_inference_debug:
            logger.debug(f"[Preprocess] Input frame shape: {frame.shape}")
        
        # Resize and pad
        img = cv2.resize(frame, (self.input_size, self.input_size))
        if enable_inference_debug:
            logger.debug(f"[Preprocess] After resize: {img.shape}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        if enable_inference_debug:
            logger.debug(f"[Preprocess] Output tensor shape: {img.shape}")
        
        return img
    
    def postprocess(self, outputs: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[Detection]:
        """Postprocess YOLO outputs to Detection objects"""
        global enable_inference_debug
        detections = []
        
        if enable_inference_debug:
            logger.debug(f"[Postprocess] Raw output shape: {outputs.shape}")
        
        # YOLOv8 output shape: (1, 84, 8400) or (1, num_classes+4, num_boxes)
        # Format: [x, y, w, h, class_scores...]
        
        output = outputs[0]  # Remove batch dimension
        if enable_inference_debug:
            logger.debug(f"[Postprocess] After removing batch: {output.shape}")
        
        # Transpose to (num_boxes, 84)
        if output.shape[0] < output.shape[1]:
            output = output.transpose()
            if enable_inference_debug:
                logger.debug(f"[Postprocess] After transpose: {output.shape}")
        
        # Extract boxes and scores
        boxes = output[:, :4]  # x, y, w, h
        scores = output[:, 4:]  # class scores
        if enable_inference_debug:
            logger.debug(f"[Postprocess] Boxes shape: {boxes.shape}, Scores shape: {scores.shape}")
        
        # Get class with highest score for each box
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        if enable_inference_debug:
            logger.debug(f"[Postprocess] Max confidences - Min: {confidences.min():.4f}, Max: {confidences.max():.4f}, Mean: {confidences.mean():.4f}")
            logger.debug(f"[Postprocess] Confidence threshold: {conf_threshold}")
            logger.debug(f"[Postprocess] Detections above threshold: {(confidences > conf_threshold).sum()} / {len(confidences)}")
        
        # Filter by confidence
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) == 0:
            if enable_inference_debug:
                logger.debug(f"[Postprocess] No detections after confidence filtering")
            return detections
        
        if enable_inference_debug:
            logger.debug(f"[Postprocess] After confidence filter: {len(boxes)} boxes remaining")
            for i in range(min(5, len(boxes))):
                logger.debug(f"  Box {i}: class_id={class_ids[i]}, conf={confidences[i]:.4f}")
        
        # Convert from center format to corner format
        # YOLO: (center_x, center_y, width, height) -> (x1, y1, x2, y2)
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )
        
        if enable_inference_debug:
            logger.debug(f"[Postprocess] NMS indices count: {len(indices)}")
        
        if len(indices) > 0:
            for i in indices.flatten():
                class_id = int(class_ids[i])
                confidence = float(confidences[i])
                
                # Get label
                if 0 <= class_id < len(self.classes):
                    label = self.classes[class_id]
                else:
                    label = f"unknown_{class_id}"
                    logger.warning(f"[Postprocess] Unknown class_id: {class_id}")
                
                # Keep bbox as normalized coordinates (0-1)
                bbox = [
                    float(x1[i]),
                    float(y1[i]),
                    float(x2[i]),
                    float(y2[i])
                ]
                
                detections.append(Detection(
                    label=label,
                    confidence=confidence,
                    bbox=bbox
                ))
                if enable_inference_debug:
                    logger.debug(f"[Postprocess] Detection added: {label} (conf={confidence:.4f})")
        
        if enable_inference_debug:
            logger.info(f"[Postprocess] Total detections returned: {len(detections)}")
        return detections
        
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on frame"""
        global enable_inference_debug
        if self.session is None or frame is None:
            if self.session is None and enable_inference_debug:
                logger.warning("[Predict] Session is None, model not loaded")
            if frame is None and enable_inference_debug:
                logger.warning("[Predict] Frame is None")
            return []
        
        try:
            # Store original dimensions
            orig_h, orig_w = frame.shape[:2]
            logger.debug(f"[Predict] Original frame size: {orig_w}x{orig_h}")
            
            # Preprocess
            input_tensor = self.preprocess(frame)
            
            # Inference
            logger.debug(f"[Predict] Running inference...")
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            logger.debug(f"[Predict] Inference complete, output shape: {[o.shape for o in outputs]}")
            
            # Postprocess
            detections = self.postprocess(outputs[0])
            
            # Convert normalized coordinates (0-1) to pixel coordinates
            # Bbox from postprocess is normalized relative to input_size (640x640)
            # Need to scale to actual frame size considering letterbox/padding
            for det in detections:
                # Normalize from 640x640 to 0-1 range
                x1_norm = det.bbox[0] / self.input_size
                y1_norm = det.bbox[1] / self.input_size
                x2_norm = det.bbox[2] / self.input_size
                y2_norm = det.bbox[3] / self.input_size
                
                # Scale to original image size
                det.bbox[0] = x1_norm * orig_w
                det.bbox[1] = y1_norm * orig_h
                det.bbox[2] = x2_norm * orig_w
                det.bbox[3] = y2_norm * orig_h
            
            if len(detections) > 0:
                logger.info(f"[Predict] ✅ Found {len(detections)} detection(s)")
            else:
                logger.debug(f"[Predict] ⚠️ No detections found")
            
            return detections
            
        except Exception as e:
            logger.error(f"[Predict] Inference error: {e}", exc_info=True)
            return []


class ConfigManager:
    """Manages configuration files and OTA updates"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.classes_file = data_dir / "classes.json"
        self.price_map_file = data_dir / "price_map.json"
        self.model_version_file = data_dir / "model_version.json"
        
    def load_classes(self) -> List[str]:
        """Load classes.json"""
        if self.classes_file.exists():
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
                logger.info(f"Loaded {len(classes)} classes")
                return classes
        logger.warning("classes.json not found, using defaults")
        return ["plate_round_large", "bowl_round_small"]
    
    def load_price_map(self) -> dict:
        """Load price_map.json"""
        if self.price_map_file.exists():
            with open(self.price_map_file, 'r', encoding='utf-8') as f:
                price_map = json.load(f)
                logger.info(f"Loaded price map with {len(price_map)} entries")
                return price_map
        logger.warning("price_map.json not found, using defaults")
        return {}
    
    def save_classes(self, classes: List[str]):
        """Save classes.json"""
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            json.dump(classes, f, indent=2)
        logger.info(f"Saved {len(classes)} classes")
    
    def save_price_map(self, price_map: dict):
        """Save price_map.json"""
        with open(self.price_map_file, 'w', encoding='utf-8') as f:
            json.dump(price_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved price map with {len(price_map)} entries")
    
    def validate_consistency(self, classes: List[str], price_map: dict) -> bool:
        """Validate that price_map keys match classes"""
        class_set = set(classes)
        price_keys = set(price_map.keys())
        
        missing = class_set - price_keys
        extra = price_keys - class_set
        
        if missing:
            logger.warning(f"Missing price map entries: {missing}")
        if extra:
            logger.warning(f"Extra price map entries: {extra}")
        
        return len(missing) == 0


# Initialize managers
config_manager = ConfigManager(DATA_DIR)
annotation_manager = AnnotationManager(DATA_DIR)
training_manager = TrainingManager(DATA_DIR, MODELS_DIR)


@app.on_event("startup")
async def startup_event():
    """Initialize backend on startup"""
    global camera, model, classes, price_map, annotation_manager, training_manager, assist_model
    
    logger.info("Starting Smart-Canteen Backend...")
    
    # Load configuration
    classes = config_manager.load_classes()
    price_map = config_manager.load_price_map()
    
    # Validate consistency
    config_manager.validate_consistency(classes, price_map)
    
    # Initialize managers
    annotation_manager = AnnotationManager(DATA_DIR)
    training_manager = TrainingManager(DATA_DIR, MODELS_DIR)
    
    # Initialize camera
    try:
        camera = CameraManager(camera_id=0)
        camera.start()
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        camera = None
    
    # Load YOLOv8n pretrained model for annotation assistance
    try:
        from ultralytics import YOLO
        assist_model = YOLO('yolov8n.pt')  # Auto-download if not exists
        logger.info("✅ Loaded YOLOv8n pretrained model for annotation assistance")
    except Exception as e:
        logger.warning(f"Failed to load assist model: {e}")
        assist_model = None
    
    # Initialize YOLO model
    model_path = MODELS_DIR / "best.onnx"
    if model_path.exists():
        model = YOLOInference(str(model_path), classes)
    else:
        logger.warning("Model not found, running without inference")
    
    logger.info("Backend startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera
    if camera:
        camera.stop()
    logger.info("Backend shutdown complete")


def generate_mjpeg_stream():
    """Generate MJPEG stream with bounding boxes"""
    global camera, model, latest_detection, is_training
    
    frame_count = 0
    
    while True:
        if not camera:
            # Send blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera Offline", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue
        
        frame = camera.read_frame()
        if frame is None:
            continue
        
        frame_count += 1
        
        # Skip inference during training
        if is_training:
            # Show training message on frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Training in Progress...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(display_frame, "Inference Paused", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue
        
        # Draw bounding boxes from latest_detection (updated by /api/detect_once)
        # No inference in video stream - it's triggered by frontend requests
        if latest_detection and len(latest_detection) > 0:
            for det in latest_detection:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['label']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream endpoint"""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/current_detection")
async def get_current_detection():
    """Get latest detection results (legacy, for backward compatibility)"""
    return JSONResponse(content={
        "detections": latest_detection,
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/detect_once")
async def detect_once():
    """
    Perform single detection on demand
    This endpoint triggers a fresh detection and returns results immediately
    """
    global latest_detection, is_training
    
    if is_training:
        return JSONResponse(content={
            "detections": [],
            "message": "Training in progress",
            "timestamp": datetime.now().isoformat()
        })
    
    if camera is None or not camera.cap or not camera.cap.isOpened():
        return JSONResponse(content={
            "detections": [],
            "error": "Camera not available",
            "timestamp": datetime.now().isoformat()
        })
    
    # Get fresh frame from camera
    frame = camera.read_frame()
    if frame is None:
        return JSONResponse(content={
            "detections": [],
            "error": "Failed to capture frame",
            "timestamp": datetime.now().isoformat()
        })
    
    # Run inference
    detections = []
    if model is not None:
        detections = model.predict(frame)
    
    # Convert Detection objects to dicts for JSON serialization
    detections_dict = [
        {
            "label": d.label,
            "confidence": d.confidence,
            "bbox": d.bbox
        }
        for d in detections
    ]
    
    # Update global latest_detection with dict format for video stream
    latest_detection = detections_dict
    
    return JSONResponse(content={
        "detections": detections_dict,
        "count": len(detections),
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/config/classes")
async def get_classes():
    """Get classes configuration"""
    return JSONResponse(content=classes)


@app.get("/api/config/price_map")
async def get_price_map():
    """Get price map configuration"""
    return JSONResponse(content=price_map)


@app.post("/api/order")
async def submit_order(order: Order, background_tasks: BackgroundTasks):
    """Submit an order"""
    # Save order locally
    orders_dir = DATA_DIR / "orders"
    orders_dir.mkdir(exist_ok=True)
    
    order_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    order_file = orders_dir / f"order_{order_id}.json"
    
    order_data = order.dict()
    order_data["order_id"] = order_id
    order_data["status"] = "pending"
    
    with open(order_file, 'w', encoding='utf-8') as f:
        json.dump(order_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Order {order_id} saved locally")
    
    # TODO: Queue for cloud upload when network is available
    
    return JSONResponse(content={
        "order_id": order_id,
        "status": "success"
    })


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "camera": camera is not None and camera.running,
        "model": model is not None,
        "timestamp": datetime.now().isoformat()
    })


# ========== Annotation & Training APIs ==========

class CaptureRequest(BaseModel):
    label: str
    bbox: Optional[List[float]] = None  # Optional [x1, y1, x2, y2]


class DeleteSampleRequest(BaseModel):
    label: str
    filename: str


@app.post("/api/annotation/capture")
async def capture_annotation(request: CaptureRequest):
    """Capture current frame as training sample"""
    global camera, classes, annotation_manager, model
    
    if not camera or not camera.frame is not None:
        raise HTTPException(status_code=400, detail="Camera not available")
    
    if request.label not in classes:
        raise HTTPException(status_code=400, detail="Invalid label")
    
    # Get current frame
    frame = camera.frame
    if frame is None:
        raise HTTPException(status_code=400, detail="No frame available")
    
    # Get class index
    class_index = classes.index(request.label)
    
    # Try to get bbox from request or detect automatically
    bbox = request.bbox
    
    # Prepare default centered square bbox (80% of min dimension)
    h, w = frame.shape[:2]
    default_bbox = get_center_square_bbox(w, h)
    
    # If no bbox provided, try to get from YOLOv8n assist model
    if bbox is None and assist_model:
        try:
            results = assist_model(frame, verbose=False, conf=0.25)
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    # Use the first detected object (usually the most prominent)
                    box = result.boxes[0]
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    logger.info(f"Using YOLOv8n detected bbox for {request.label}: {bbox}")
        except Exception as e:
            logger.warning(f"Could not get YOLOv8n detection bbox: {e}")
    
    # Final fallback: centered square bbox to keep annotations consistent
    if bbox is None:
        bbox = default_bbox
        logger.info(f"Using default centered square bbox for {request.label}: {bbox}")
    
    # Save annotation with bbox
    filename = annotation_manager.capture_sample(frame, request.label, class_index, bbox)
    
    # Get updated stats
    stats = annotation_manager.get_stats(classes)
    total_count = stats[request.label]["count"]
    
    return JSONResponse(content={
        "success": True,
        "filename": filename,
        "label": request.label,
        "total_count": total_count
    })


@app.get("/api/annotation/stats")
async def get_annotation_stats():
    """Get annotation statistics"""
    global classes, annotation_manager
    
    stats = annotation_manager.get_stats(classes)
    return JSONResponse(content=stats)


@app.get("/api/annotation/preview")
async def get_annotation_preview():
    """Get current detections for annotation preview using YOLOv8n pretrained model"""
    global camera, assist_model, is_training
    
    if not camera or camera.frame is None:
        raise HTTPException(status_code=400, detail="Camera not available")
    
    frame = camera.frame
    if frame is None:
        raise HTTPException(status_code=400, detail="No frame available")
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    default_detection = {
        "class_id": -1,
        "class_name": "center_focus",
        "confidence": 1.0,
        "bbox": [float(x) for x in get_center_square_bbox(w, h)],
        "is_default": True
    }
    
    # Use YOLOv8n pretrained model for stable object detection
    if assist_model:
        try:
            # Run YOLOv8 inference
            results = assist_model(frame, verbose=False, conf=0.25)
            
            # Format detections for frontend
            boxes = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bbox coordinates
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get COCO class name
                        cls_name = result.names[cls_id] if cls_id < len(result.names) else f"object_{cls_id}"
                        
                        boxes.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                        })
            
            return JSONResponse(content={
                "success": True,
                "frame_width": w,
                "frame_height": h,
                "detections": boxes if len(boxes) > 0 else [default_detection]
            })
        except Exception as e:
            logger.error(f"Preview detection error: {e}")
            return JSONResponse(content={
                "success": True,
                "frame_width": w,
                "frame_height": h,
                "detections": [default_detection]
            })
    
    return JSONResponse(content={
        "success": True,
        "frame_width": w,
        "frame_height": h,
        "detections": [default_detection]
    })


@app.delete("/api/annotation/sample")
async def delete_annotation_sample(request: DeleteSampleRequest):
    """Delete an annotation sample"""
    global annotation_manager
    
    success = annotation_manager.delete_sample(request.filename)
    
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    return JSONResponse(content={"success": True})


@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Start model training in background"""
    global classes, training_manager, is_training, model
    
    if training_manager.status == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Pause inference during training
    is_training = True
    enable_inference_debug = False
    logger.info("🚫 Inference paused for training, debug logs disabled")
    
    # Start training in background thread
    def train_in_background():
        global is_training, model, enable_inference_debug
        try:
            training_manager.train_model(classes, epochs=50, batch_size=8)
            
            # Reload model after training
            logger.info("🔄 Reloading model after training...")
            model_path = MODELS_DIR / "best.onnx"
            if model_path.exists():
                try:
                    model = YOLOInference(str(model_path), classes)
                    logger.info("✅ Model reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload model: {e}")
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
        finally:
            # Resume inference after training
            is_training = False
            enable_inference_debug = True
            logger.info("✅ Inference resumed, debug logs enabled")
    
    thread = threading.Thread(target=train_in_background, daemon=True)
    thread.start()
    
    task_id = training_manager.current_task
    
    return JSONResponse(content={
        "task_id": task_id,
        "status": "started"
    })


@app.get("/api/training/progress/{task_id}")
async def get_training_progress(task_id: str):
    """Get training progress"""
    global training_manager
    
    progress_info = training_manager.get_progress(task_id)
    return JSONResponse(content=progress_info)


# Serve frontend static files (if built)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
