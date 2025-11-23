# YOLO Training Guide for Animal Behavior Classification

## Overview

This guide explains how to train a YOLO model to classify animal behaviors (scratching, pacing, sleeping) and get time percentages for each behavior class.

## Training Strategy

### Option 1: YOLOv8 with Video Classification (Recommended)

YOLOv8 can be adapted for video classification by:
1. **Frame-by-frame detection**: Process each frame and classify behavior
2. **Temporal aggregation**: Aggregate predictions over time to get percentages

### Option 2: YOLOv8 + Action Recognition

Use YOLOv8 for object detection, then add a temporal model for action recognition.

---

## Step-by-Step Training Process

### 1. Data Collection & Preparation

#### Collect Training Videos
- **Scratching**: Videos of animals scratching themselves
- **Pacing**: Videos of animals pacing back and forth
- **Sleeping**: Videos of animals sleeping/resting
- **Other behaviors**: (optional) walking, eating, etc.

**Recommended:**
- Minimum 50-100 videos per class
- Videos should be 10-60 seconds long
- Various lighting conditions, angles, and animal species
- Mix of healthy and unhealthy behaviors

#### Organize Dataset Structure
```
data/
├── training/
│   ├── scratching/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── pacing/
│   │   ├── video1.mp4
│   │   └── ...
│   └── sleeping/
│       ├── video1.mp4
│       └── ...
├── validation/
│   ├── scratching/
│   ├── pacing/
│   └── sleeping/
└── test/
    ├── scratching/
    ├── pacing/
    └── sleeping/
```

### 2. Annotation Strategy

#### Option A: Frame-Level Annotation (More Accurate)
- Extract frames from videos (every 1-2 seconds)
- Label each frame with behavior class
- Creates many training samples

#### Option B: Video-Level Annotation (Faster)
- Label entire video with primary behavior
- Faster to annotate, but less precise

#### Option C: Temporal Annotation (Best for Percentages)
- Use video annotation tools (e.g., CVAT, Label Studio)
- Mark time segments: `[0-10s: pacing, 10-20s: scratching, 20-30s: pacing]`
- This directly gives you time percentages!

**Recommended Tool: CVAT (Computer Vision Annotation Tool)**
```bash
# Install CVAT
docker-compose up -d
# Access at http://localhost:8080
```

### 3. Training Setup

#### Install YOLOv8
```bash
pip install ultralytics
```

#### Prepare Dataset in YOLO Format

**For Frame-Based Training:**
```python
# Extract frames and create YOLO dataset
import cv2
import os

def extract_frames_for_yolo(video_path, output_dir, class_id, fps_interval=1):
    """
    Extract frames from video for YOLO training.
    
    Args:
        video_path: Path to video
        output_dir: Where to save frames
        class_id: Behavior class (0=scratching, 1=pacing, 2=sleeping)
        fps_interval: Extract every N seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame
            frame_path = os.path.join(output_dir, f"{class_id}_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Create label file (for classification, we use image-level labels)
            # YOLOv8 classification format
            label_path = frame_path.replace('.jpg', '.txt')
            with open(label_path, 'w') as f:
                f.write(str(class_id))
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
```

**Dataset Structure:**
```
yolo_dataset/
├── train/
│   ├── images/
│   │   ├── 0_000001.jpg  # scratching
│   │   ├── 1_000001.jpg  # pacing
│   │   └── 2_000001.jpg  # sleeping
│   └── labels/
│       ├── 0_000001.txt  # contains "0"
│       ├── 1_000001.txt  # contains "1"
│       └── 2_000001.txt  # contains "2"
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml:**
```yaml
path: /path/to/yolo_dataset
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: scratching
  1: pacing
  2: sleeping

nc: 3  # number of classes
```

### 4. Training Script

```python
from ultralytics import YOLO
import torch

# Initialize model
model = YOLO('yolov8n-cls.pt')  # or yolov8s-cls.pt, yolov8m-cls.pt

# Train
results = model.train(
    data='/path/to/yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    project='behavior_classification',
    name='yolov8_behavior',
    patience=20,  # Early stopping
    save=True,
    plots=True
)

# Validate
metrics = model.val()
print(f"Accuracy: {metrics.top1}")
```

### 5. Inference for Time Percentages

```python
import cv2
from ultralytics import YOLO
from collections import Counter

def analyze_video_behavior_percentages(video_path, model_path):
    """
    Analyze video and return time percentages for each behavior.
    
    Returns:
        {
            "scratching": 0.25,  # 25% of time
            "pacing": 0.60,      # 60% of time
            "sleeping": 0.15     # 15% of time
        }
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process every N frames (e.g., every 1 second)
    frame_interval = int(fps)  # 1 second
    predictions = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Run inference
            results = model(frame)
            
            # Get top prediction
            probs = results[0].probs
            top_class = probs.top1
            confidence = probs.top1conf.item()
            
            # Map class ID to behavior name
            class_names = {0: "scratching", 1: "pacing", 2: "sleeping"}
            behavior = class_names[top_class]
            
            predictions.append(behavior)
        
        frame_count += 1
    
    cap.release()
    
    # Calculate percentages
    total_predictions = len(predictions)
    if total_predictions == 0:
        return {"scratching": 0.0, "pacing": 0.0, "sleeping": 0.0}
    
    behavior_counts = Counter(predictions)
    percentages = {
        "scratching": behavior_counts.get("scratching", 0) / total_predictions,
        "pacing": behavior_counts.get("pacing", 0) / total_predictions,
        "sleeping": behavior_counts.get("sleeping", 0) / total_predictions
    }
    
    # Normalize to ensure sum = 1.0
    total = sum(percentages.values())
    if total > 0:
        percentages = {k: v / total for k, v in percentages.items()}
    
    return percentages
```

---

## Alternative: Using Pre-trained Models + Fine-tuning

### Option: Use YOLOv8 Classification Model

1. **Start with pre-trained YOLOv8 classification model**
2. **Fine-tune on your behavior dataset**
3. **Much faster than training from scratch**

```python
from ultralytics import YOLO

# Load pre-trained classification model
model = YOLO('yolov8n-cls.pt')  # or yolov8s-cls.pt for better accuracy

# Fine-tune on your dataset
model.train(
    data='path/to/your/dataset',
    epochs=50,  # Fewer epochs needed for fine-tuning
    imgsz=224,
    batch=32,
    lr0=0.001,  # Lower learning rate for fine-tuning
    pretrained=True
)
```

---

## Training Tips

### 1. Data Augmentation
- Random rotations, flips, brightness adjustments
- Helps model generalize better

### 2. Class Balance
- Ensure roughly equal number of samples per class
- Use data augmentation if one class has fewer samples

### 3. Validation Split
- Use 80% train, 10% validation, 10% test
- Monitor validation accuracy to prevent overfitting

### 4. Hyperparameter Tuning
- Learning rate: Start with 0.01, adjust based on loss
- Batch size: Larger = faster training, but needs more GPU memory
- Image size: 224x224 for classification (faster), 640x640 for detection

### 5. Early Stopping
- Stop training if validation accuracy doesn't improve for 20 epochs
- Prevents overfitting

---

## Quick Start Script

Save this as `train_yolo.py`:

```python
"""
Quick start script for training YOLO behavior classifier.
"""
from ultralytics import YOLO
import os

# Configuration
DATA_YAML = "data/yolo_dataset/data.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 224
MODEL_SIZE = "n"  # n=nano, s=small, m=medium, l=large, x=xlarge

def main():
    print("Starting YOLO training for behavior classification...")
    
    # Check if dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"Error: Dataset not found at {DATA_YAML}")
        print("Please prepare your dataset first.")
        return
    
    # Load model
    model_name = f"yolov8{MODEL_SIZE}-cls.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print("Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="behavior_classification",
        name="yolov8_behavior",
        patience=20,
        save=True,
        plots=True
    )
    
    print("\nTraining complete!")
    print(f"Best model saved at: {results.save_dir}")
    
    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1:.2f}%")
    print(f"Top-5 Accuracy: {metrics.top5:.2f}%")

if __name__ == "__main__":
    main()
```

---

## Integration with Backend

Once trained, save your model and integrate it into the backend. The model will:
1. Process video frame-by-frame
2. Classify each frame's behavior
3. Calculate time percentages
4. Return percentages to the API

See `backend/app.py` for integration code.

---

## Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **CVAT Annotation Tool**: https://github.com/openvinotoolkit/cvat
- **Label Studio**: https://labelstud.io/
- **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics

---

## Next Steps

1. Collect and organize your training videos
2. Annotate videos (use CVAT or similar tool)
3. Extract frames and create YOLO dataset
4. Train model using the script above
5. Validate model accuracy
6. Integrate into backend (see updated `backend/app.py`)

