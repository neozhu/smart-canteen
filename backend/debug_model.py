"""
Debug script to test ONNX model inference
"""
import json
import cv2
import numpy as np
from pathlib import Path

def test_onnx_model():
    """Test ONNX model with sample image"""
    
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "best.onnx"
    classes_path = base_dir / "data" / "classes.json"
    dataset_dir = base_dir / "data" / "dataset" / "images"
    
    print("=" * 60)
    print("ONNX Model Debug Test")
    print("=" * 60)
    
    # Check files
    print(f"\n1. Checking files...")
    print(f"   Model exists: {model_path.exists()} - {model_path}")
    print(f"   Classes exists: {classes_path.exists()} - {classes_path}")
    print(f"   Dataset exists: {dataset_dir.exists()} - {dataset_dir}")
    
    if not model_path.exists():
        print("\n❌ Model file not found!")
        return
    
    # Load classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    print(f"\n2. Classes loaded: {len(classes)}")
    for i, cls in enumerate(classes):
        print(f"   [{i}] {cls}")
    
    # Load ONNX model
    print(f"\n3. Loading ONNX model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # Model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"   ✅ Model loaded successfully")
        print(f"   Input name: {input_info.name}")
        print(f"   Input shape: {input_info.shape}")
        print(f"   Output name: {output_info.name}")
        print(f"   Output shape: {output_info.shape}")
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Find a test image
    print(f"\n4. Looking for test images...")
    if dataset_dir.exists():
        test_images = list(dataset_dir.glob("*.jpg"))[:3]
        if test_images:
            print(f"   Found {len(test_images)} test image(s)")
        else:
            print(f"   ⚠️ No images found in dataset, will use camera")
            test_images = []
    else:
        test_images = []
    
    # Try camera if no test images
    if not test_images:
        print(f"\n   Trying camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"   ✅ Got frame from camera: {frame.shape}")
                test_frame = frame
            else:
                print(f"   ❌ Failed to read from camera")
                return
        else:
            print(f"   ❌ Failed to open camera")
            return
    else:
        # Use first test image
        test_image_path = test_images[0]
        print(f"   Using: {test_image_path.name}")
        test_frame = cv2.imread(str(test_image_path))
        print(f"   Image shape: {test_frame.shape}")
    
    # Preprocess
    print(f"\n5. Preprocessing...")
    img = cv2.resize(test_frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    print(f"   Input tensor shape: {img.shape}")
    print(f"   Input tensor dtype: {img.dtype}")
    print(f"   Input tensor range: [{img.min():.4f}, {img.max():.4f}]")
    
    # Run inference
    print(f"\n6. Running inference...")
    try:
        outputs = session.run([output_info.name], {input_info.name: img})
        output = outputs[0]
        print(f"   ✅ Inference successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return
    
    # Parse output
    print(f"\n7. Parsing output...")
    
    output = output[0]  # Remove batch
    print(f"   After removing batch: {output.shape}")
    
    # Transpose if needed
    if output.shape[0] < output.shape[1]:
        output = output.transpose()
        print(f"   After transpose: {output.shape}")
    
    boxes = output[:, :4]
    scores = output[:, 4:]
    
    print(f"   Boxes: {boxes.shape}")
    print(f"   Scores: {scores.shape}")
    
    # Get predictions
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    print(f"\n8. Confidence statistics:")
    print(f"   Min confidence: {confidences.min():.4f}")
    print(f"   Max confidence: {confidences.max():.4f}")
    print(f"   Mean confidence: {confidences.mean():.4f}")
    print(f"   Median confidence: {np.median(confidences):.4f}")
    
    # Check different thresholds
    print(f"\n9. Detection counts at different thresholds:")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        count = (confidences > threshold).sum()
        print(f"   > {threshold:.2f}: {count} detections")
        if count > 0 and count <= 10:
            # Show top detections at this threshold
            top_indices = np.where(confidences > threshold)[0][:5]
            for idx in top_indices:
                cls_id = class_ids[idx]
                cls_name = classes[cls_id] if cls_id < len(classes) else f"unknown_{cls_id}"
                print(f"      - {cls_name}: {confidences[idx]:.4f}")
    
    # Show top 10 predictions
    print(f"\n10. Top 10 predictions (regardless of threshold):")
    top_10_indices = np.argsort(confidences)[-10:][::-1]
    for rank, idx in enumerate(top_10_indices, 1):
        cls_id = class_ids[idx]
        cls_name = classes[cls_id] if cls_id < len(classes) else f"unknown_{cls_id}"
        print(f"   {rank}. {cls_name}: {confidences[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("Debug test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_onnx_model()
