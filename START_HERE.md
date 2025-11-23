# 🐕 Start Here: Dog Distress Detection Training

## Quick Start Guide

### Step 1: Upload Your Videos

Upload your dog behavior videos to these folders:

```
data/dog_training/
├── train/          ← Upload 80% of videos here
│   ├── pacing/     ← Dogs pacing (distress)
│   ├── scratching/ ← Dogs scratching (distress)
│   ├── sleeping/   ← Dogs sleeping (normal)
│   ├── walking/    ← Dogs walking (normal)
│   └── resting/    ← Dogs resting (normal)
├── val/            ← Upload 10% of videos here (same structure)
└── test/           ← Upload 10% of videos here (same structure)
```

**What to upload:**
- **Distressed behaviors**: Videos of dogs pacing or scratching excessively
- **Normal behaviors**: Videos of dogs sleeping, walking normally, or resting calmly

**Requirements:**
- Video formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Video length: 10-60 seconds
- Minimum: 50-100 videos per behavior class (more is better!)

---

### Step 2: Run Training Pipeline

Once videos are uploaded, run:

```bash
# Option 1: Run all steps automatically
./scripts/run_all_steps.sh

# Option 2: Run steps manually (see below)
```

**Manual Steps:**

```bash
# 1. Extract frames from videos
python scripts/extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing
python scripts/extract_frames.py data/dog_training/train/scratching data/dog_frames/train/scratching
python scripts/extract_frames.py data/dog_training/train/sleeping data/dog_frames/train/sleeping
python scripts/extract_frames.py data/dog_training/train/walking data/dog_frames/train/walking
python scripts/extract_frames.py data/dog_training/train/resting data/dog_frames/train/resting

# Repeat for val/ and test/ splits...

# 2. Prepare YOLO dataset
python scripts/prepare_yolo_dataset.py data/dog_frames/train data/yolo_dataset/train
python scripts/prepare_yolo_dataset.py data/dog_frames/val data/yolo_dataset/val

# 3. Train model
python scripts/train_dog_behavior.py
```

---

### Step 3: Test Your Model

```bash
python scripts/test_dog_model.py <video_path> dog_behavior_classification/yolov8_dog_behavior/weights/best.pt
```

---

### Step 4: Use in Backend

```bash
# Set model path
export YOLO_MODEL_PATH="dog_behavior_classification/yolov8_dog_behavior/weights/best.pt"

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run backend
cd backend
python app.py
```

---

## Directory Structure

All directories are already created! Just upload your videos:

```
data/
├── dog_training/        ← UPLOAD VIDEOS HERE
│   ├── train/
│   ├── val/
│   └── test/
├── dog_frames/         ← Auto-generated (frames extracted from videos)
├── yolo_dataset/        ← Auto-generated (prepared for YOLO)
└── test_videos/        ← Put test videos here (optional)
```

---

## What Happens During Training?

1. **Frame Extraction**: Videos → Individual frames (1 frame per second)
2. **Dataset Preparation**: Frames organized for YOLO format
3. **Training**: YOLO learns to classify behaviors (1-4 hours)
4. **Model Saved**: Best model saved automatically

---

## Troubleshooting

**"No videos found"**
- Make sure videos are in the correct folders
- Check video formats (.mp4, .avi, .mov, .mkv)

**"Dataset not found"**
- Run frame extraction first
- Then run dataset preparation

**Low accuracy**
- Collect more training videos
- Ensure balanced classes (similar number per behavior)
- Use better quality videos

**Out of memory**
- Reduce batch size in `scripts/train_dog_behavior.py` (change `BATCH_SIZE=8`)

---

## Need More Help?

- **Detailed Guide**: See `DOG_DISTRESS_TRAINING_GUIDE.md`
- **Quick Reference**: See `QUICK_START_DOG_TRAINING.md`
- **Scripts**: See `scripts/README.md`

---

## Ready to Start?

1. ✅ Upload videos to `data/dog_training/`
2. ✅ Run `./scripts/run_all_steps.sh`
3. ✅ Wait for training to complete
4. ✅ Test your model
5. ✅ Use in backend!

Good luck! 🐕

