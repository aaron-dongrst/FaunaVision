# Training Scripts

Scripts for training YOLO on dog behavior videos.

## Scripts:

### 1. `extract_frames.py`
Extracts frames from videos for training.

**Usage:**
```bash
python scripts/extract_frames.py <video_dir> <output_dir> [fps_interval]
```

**Example:**
```bash
python scripts/extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing 1.0
```

### 2. `prepare_yolo_dataset.py`
Prepares extracted frames for YOLO training format.

**Usage:**
```bash
python scripts/prepare_yolo_dataset.py <frames_dir> <output_dir>
```

**Example:**
```bash
python scripts/prepare_yolo_dataset.py data/dog_frames/train data/yolo_dataset/train
```

### 3. `train_dog_behavior.py`
Trains the YOLO model on your dataset.

**Usage:**
```bash
python scripts/train_dog_behavior.py
```

**Requirements:**
- `data/yolo_dataset/data.yaml` must exist
- Training and validation datasets must be prepared

### 4. `test_dog_model.py`
Tests a trained model on a video.

**Usage:**
```bash
python scripts/test_dog_model.py <video_path> <model_path>
```

**Example:**
```bash
python scripts/test_dog_model.py data/test_videos/dog.mp4 dog_behavior_classification/yolov8_dog_behavior/weights/best.pt
```

## Workflow:

1. Upload videos to `data/dog_training/`
2. Extract frames: Run `extract_frames.py` for each behavior
3. Prepare dataset: Run `prepare_yolo_dataset.py` for train and val
4. Train model: Run `train_dog_behavior.py`
5. Test model: Run `test_dog_model.py`

See `DOG_DISTRESS_TRAINING_GUIDE.md` for detailed instructions.

