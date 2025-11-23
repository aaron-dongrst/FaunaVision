# Complete Guide: Training YOLO for Dog Distress Detection

This guide walks you through training a YOLO model to detect dog behaviors and determine if dogs are distressed.

## Overview

We'll train YOLO to classify dog behaviors (pacing, scratching, sleeping, etc.), then use behavior percentages to determine distress levels. Dogs showing excessive pacing, repetitive scratching, or other stress behaviors will be flagged as distressed.

---

## Step 1: Collect and Organize Your Dog Videos

### 1.1 Gather Training Videos

You need videos of dogs showing different behaviors. Collect:

**Distressed Behaviors:**
- Pacing back and forth
- Excessive scratching/licking
- Whining/barking (if visible)
- Restless movement
- Repetitive behaviors

**Normal/Healthy Behaviors:**
- Sleeping/resting calmly
- Walking normally
- Playing
- Eating
- Relaxed behavior

**Recommended:**
- Minimum 50-100 videos per behavior class
- Videos should be 10-60 seconds long
- Various lighting conditions, angles, and dog breeds
- Mix of indoor and outdoor settings

### 1.2 Organize Your Dataset

Create this folder structure:

```bash
cd /Users/aarondong/Desktop/Faunavision
mkdir -p data/dog_training/{train,val,test}/{pacing,scratching,sleeping,walking,resting}
```

**Folder Structure:**
```
data/dog_training/
├── train/
│   ├── pacing/          # Dogs pacing (distress indicator)
│   ├── scratching/      # Dogs scratching excessively (distress indicator)
│   ├── sleeping/        # Dogs sleeping (normal)
│   ├── walking/         # Dogs walking normally (normal)
│   └── resting/         # Dogs resting calmly (normal)
├── val/                 # Same structure for validation
└── test/                # Same structure for testing
```

**Split your videos:**
- 80% → `train/`
- 10% → `val/`
- 10% → `test/`

---

## Step 2: Extract Frames from Videos

YOLO classification models work on images, so we need to extract frames from videos.

### 2.1 Create Frame Extraction Script

Create `scripts/extract_frames.py`:

```python
"""
Extract frames from dog behavior videos for YOLO training.
"""
import cv2
import os
from pathlib import Path

def extract_frames(video_dir, output_dir, fps_interval=1.0):
    """
    Extract frames from videos in video_dir and save to output_dir.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Where to save extracted frames
        fps_interval: Extract every N seconds (default: 1.0)
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob("*.mp4")) + \
                  list(video_dir.glob("*.avi")) + \
                  list(video_dir.glob("*.mov"))
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * fps_interval) if fps > 0 else 1
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_filename = f"{video_path.stem}_{saved_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"  Extracted {saved_count} frames")
    
    print(f"\nDone! Frames saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video_dir> <output_dir> [fps_interval]")
        print("\nExample:")
        print("  python extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing 1.0")
        sys.exit(1)
    
    video_dir = sys.argv[1]
    output_dir = sys.argv[2]
    fps_interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    extract_frames(video_dir, output_dir, fps_interval)
```

### 2.2 Extract Frames for Each Behavior

Run this for each behavior class:

```bash
# Create frames directory
mkdir -p data/dog_frames/{train,val,test}/{pacing,scratching,sleeping,walking,resting}

# Extract frames for training set
python scripts/extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing
python scripts/extract_frames.py data/dog_training/train/scratching data/dog_frames/train/scratching
python scripts/extract_frames.py data/dog_training/train/sleeping data/dog_frames/train/sleeping
python scripts/extract_frames.py data/dog_training/train/walking data/dog_frames/train/walking
python scripts/extract_frames.py data/dog_training/train/resting data/dog_frames/train/resting

# Extract frames for validation set
python scripts/extract_frames.py data/dog_training/val/pacing data/dog_frames/val/pacing
python scripts/extract_frames.py data/dog_training/val/scratching data/dog_frames/val/scratching
python scripts/extract_frames.py data/dog_training/val/sleeping data/dog_frames/val/sleeping
python scripts/extract_frames.py data/dog_training/val/walking data/dog_frames/val/walking
python scripts/extract_frames.py data/dog_training/val/resting data/dog_frames/val/resting

# Extract frames for test set
python scripts/extract_frames.py data/dog_training/test/pacing data/dog_frames/test/pacing
python scripts/extract_frames.py data/dog_training/test/scratching data/dog_frames/test/scratching
python scripts/extract_frames.py data/dog_training/test/sleeping data/dog_frames/test/sleeping
python scripts/extract_frames.py data/dog_training/test/walking data/dog_frames/test/walking
python scripts/extract_frames.py data/dog_training/test/resting data/dog_frames/test/resting
```

---

## Step 3: Create YOLO Dataset Structure

YOLO classification needs a specific structure with labels.

### 3.1 Create Dataset Preparation Script

Create `scripts/prepare_yolo_dataset.py`:

```python
"""
Prepare dog behavior frames for YOLO classification training.
"""
import os
import shutil
from pathlib import Path

# Behavior classes (mapped to numbers)
BEHAVIOR_CLASSES = {
    "pacing": 0,
    "scratching": 1,
    "sleeping": 2,
    "walking": 3,
    "resting": 4
}

def prepare_yolo_dataset(frames_dir, output_dir):
    """
    Prepare frames for YOLO classification format.
    
    YOLO classification format:
    - images/ folder with all images
    - labels/ folder with .txt files (one line: class_id)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each behavior class
    for behavior_name, class_id in BEHAVIOR_CLASSES.items():
        behavior_frames_dir = frames_dir / behavior_name
        
        if not behavior_frames_dir.exists():
            print(f"Warning: {behavior_frames_dir} not found, skipping...")
            continue
        
        print(f"Processing {behavior_name} (class {class_id})...")
        
        frame_files = list(behavior_frames_dir.glob("*.jpg")) + \
                     list(behavior_frames_dir.glob("*.png"))
        
        for frame_file in frame_files:
            # Copy image
            new_image_name = f"{behavior_name}_{frame_file.name}"
            new_image_path = images_dir / new_image_name
            shutil.copy2(frame_file, new_image_path)
            
            # Create label file
            label_name = new_image_name.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = labels_dir / label_name
            
            with open(label_path, 'w') as f:
                f.write(str(class_id))
        
        print(f"  Processed {len(frame_files)} frames")
    
    print(f"\nDataset prepared at: {output_dir}")
    print(f"Total images: {len(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python prepare_yolo_dataset.py <frames_dir> <output_dir>")
        print("\nExample:")
        print("  python prepare_yolo_dataset.py data/dog_frames/train data/yolo_dataset/train")
        sys.exit(1)
    
    frames_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    prepare_yolo_dataset(frames_dir, output_dir)
```

### 3.2 Prepare Training and Validation Datasets

```bash
# Create YOLO dataset directories
mkdir -p data/yolo_dataset/{train,val}

# Prepare training dataset
python scripts/prepare_yolo_dataset.py data/dog_frames/train data/yolo_dataset/train

# Prepare validation dataset
python scripts/prepare_yolo_dataset.py data/dog_frames/val data/yolo_dataset/val
```

### 3.3 Create data.yaml Configuration

Create `data/yolo_dataset/data.yaml`:

```yaml
path: /Users/aarondong/Desktop/Faunavision/data/yolo_dataset
train: train/images
val: val/images
test: test/images

# Dog behavior classes
names:
  0: pacing
  1: scratching
  2: sleeping
  3: walking
  4: resting

nc: 5  # number of classes
```

---

## Step 4: Install YOLO and Dependencies

```bash
# Install ultralytics (YOLO)
pip install ultralytics

# Verify installation
python -c "from ultralytics import YOLO; print('YOLO installed successfully!')"
```

---

## Step 5: Train the YOLO Model

### 5.1 Create Training Script

Create `scripts/train_dog_behavior.py`:

```python
"""
Train YOLO model for dog behavior classification.
"""
from ultralytics import YOLO
import torch
import os

# Configuration
DATA_YAML = "data/yolo_dataset/data.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 224
MODEL_SIZE = "n"  # n=nano, s=small, m=medium, l=large, x=xlarge

def main():
    print("="*60)
    print("YOLO Dog Behavior Classification Training")
    print("="*60)
    print()
    
    # Check if dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"Error: Dataset not found at {DATA_YAML}")
        print("Please prepare your dataset first.")
        return
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    # Load pre-trained model
    model_name = f"yolov8{MODEL_SIZE}-cls.pt"
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print("\nStarting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Data: {DATA_YAML}")
    print()
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project="dog_behavior_classification",
        name="yolov8_dog_behavior",
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best model saved at: {results.save_dir}")
    
    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"  Top-1 Accuracy: {metrics.top1:.2f}%")
    print(f"  Top-5 Accuracy: {metrics.top5:.2f}%")
    
    # Save model path
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"\nBest model path: {best_model_path}")
    print("\nTo use this model, set:")
    print(f"  export YOLO_MODEL_PATH=\"{best_model_path}\"")

if __name__ == "__main__":
    main()
```

### 5.2 Run Training

```bash
# Make sure you're in the project root
cd /Users/aarondong/Desktop/Faunavision

# Run training
python scripts/train_dog_behavior.py
```

**Training will:**
- Take 1-4 hours depending on your GPU and dataset size
- Save checkpoints periodically
- Show progress with accuracy metrics
- Save the best model automatically

**Expected Output:**
```
Training complete!
Best model saved at: dog_behavior_classification/yolov8_dog_behavior/weights/best.pt
Top-1 Accuracy: 85.23%
```

---

## Step 6: Update Backend for Dog Distress Detection

### 6.1 Update Behavior Classes

Edit `src/yolo_behavior_classifier.py`:

```python
# Update behavior classes for dogs
self.behavior_classes = {
    0: "pacing",
    1: "scratching",
    2: "sleeping",
    3: "walking",
    4: "resting"
}

# Define distress indicators
self.distress_behaviors = ["pacing", "scratching"]  # These indicate distress
```

### 6.2 Update Distress Detection Logic

Add a method to determine distress level:

```python
def calculate_distress_level(self, percentages: Dict[str, float]) -> Dict:
    """
    Calculate distress level based on behavior percentages.
    
    Args:
        percentages: Behavior time percentages
        
    Returns:
        Dictionary with distress assessment
    """
    distress_behaviors = ["pacing", "scratching"]
    normal_behaviors = ["sleeping", "walking", "resting"]
    
    distress_percentage = sum(
        percentages.get(behavior, 0.0) 
        for behavior in distress_behaviors
    )
    
    normal_percentage = sum(
        percentages.get(behavior, 0.0) 
        for behavior in normal_behaviors
    )
    
    # Determine distress level
    if distress_percentage > 0.5:  # More than 50% distress behaviors
        distress_level = "high"
    elif distress_percentage > 0.3:  # More than 30% distress behaviors
        distress_level = "moderate"
    elif distress_percentage > 0.1:  # More than 10% distress behaviors
        distress_level = "low"
    else:
        distress_level = "none"
    
    return {
        "distress_level": distress_level,
        "distress_percentage": distress_percentage,
        "normal_percentage": normal_percentage,
        "is_distressed": distress_percentage > 0.3  # Threshold for distress
    }
```

---

## Step 7: Configure and Test

### 7.1 Set Environment Variables

```bash
# Set path to your trained model
export YOLO_MODEL_PATH="dog_behavior_classification/yolov8_dog_behavior/weights/best.pt"

# Set API key (OpenAI or Gemini)
export OPENAI_API_KEY="your-key-here"
# OR
export GEMINI_API_KEY="your-key-here"
export USE_GEMINI=true

# Optional
export PORT=5000
```

### 7.2 Test the Model

Create `scripts/test_dog_model.py`:

```python
"""
Test trained YOLO model on a dog video.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.yolo_behavior_classifier import YOLOBehaviorClassifier

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_dog_model.py <video_path> <model_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2]
    
    print("Loading YOLO model...")
    classifier = YOLOBehaviorClassifier(model_path=model_path)
    
    print(f"\nAnalyzing video: {video_path}")
    percentages = classifier.analyze_video_percentages(video_path)
    
    print("\nBehavior Percentages:")
    for behavior, percentage in percentages.items():
        print(f"  {behavior.capitalize()}: {percentage:.1%}")
    
    # Calculate distress
    distress_result = classifier.calculate_distress_level(percentages)
    
    print(f"\nDistress Assessment:")
    print(f"  Distress Level: {distress_result['distress_level']}")
    print(f"  Distress Percentage: {distress_result['distress_percentage']:.1%}")
    print(f"  Is Distressed: {distress_result['is_distressed']}")

if __name__ == "__main__":
    main()
```

Test with a video:

```bash
python scripts/test_dog_model.py data/test_videos/dog_pacing.mp4 dog_behavior_classification/yolov8_dog_behavior/weights/best.pt
```

---

## Step 8: Run the Backend

```bash
cd backend
python app.py
```

The backend will:
1. Load your trained YOLO model
2. Process videos to get behavior percentages
3. Send to OpenAI/Gemini for health assessment
4. Return distress assessment

---

## Step 9: Test End-to-End

### 9.1 Test with Frontend

1. Start frontend: `cd frontend && npm start`
2. Upload a dog video
3. Enter parameters (species: "dog", etc.)
4. Click "Analyze"
5. View behavior percentages and distress assessment

### 9.2 Test with API

```bash
curl -X POST http://localhost:5000/analyze \
  -F "video=@data/test_videos/dog_pacing.mp4" \
  -F "species=dog" \
  -F "age=3 years" \
  -F "diet=commercial dog food"
```

---

## Troubleshooting

### Low Accuracy
- **Solution**: Collect more training data, especially for underrepresented classes
- **Solution**: Use data augmentation (already enabled in YOLO)
- **Solution**: Train for more epochs

### Model Not Loading
- **Check**: `YOLO_MODEL_PATH` environment variable is set correctly
- **Check**: Model file exists at the path
- **Check**: Model file is valid (not corrupted)

### Out of Memory
- **Solution**: Reduce batch size (e.g., `BATCH_SIZE=8`)
- **Solution**: Use smaller model (e.g., `yolov8n-cls.pt` instead of `yolov8m-cls.pt`)
- **Solution**: Reduce image size (e.g., `IMAGE_SIZE=128`)

### Percentages Don't Make Sense
- **Check**: Video quality and lighting
- **Check**: Model accuracy (run validation)
- **Check**: Frame interval (try different values)

---

## Next Steps

1. **Collect More Data**: More videos = better accuracy
2. **Fine-tune Thresholds**: Adjust distress thresholds based on your data
3. **Add More Behaviors**: Add classes like "barking", "whining", etc.
4. **Deploy**: Set up production environment

---

## Quick Reference

**Key Files:**
- Training script: `scripts/train_dog_behavior.py`
- YOLO classifier: `src/yolo_behavior_classifier.py`
- Backend API: `backend/app.py`
- Dataset config: `data/yolo_dataset/data.yaml`

**Key Commands:**
```bash
# Extract frames
python scripts/extract_frames.py <video_dir> <output_dir>

# Prepare dataset
python scripts/prepare_yolo_dataset.py <frames_dir> <output_dir>

# Train model
python scripts/train_dog_behavior.py

# Test model
python scripts/test_dog_model.py <video> <model>

# Run backend
cd backend && python app.py
```

Good luck with your training! 🐕

