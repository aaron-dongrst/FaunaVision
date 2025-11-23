# Dog Training Videos Directory

## Purpose
Upload your dog behavior videos here for training the YOLO model.

## Folder Structure

```
dog_training/
├── train/          # 80% of your videos go here
│   ├── pacing/     # Videos of dogs pacing (distress behavior)
│   ├── scratching/ # Videos of dogs scratching excessively (distress behavior)
│   ├── sleeping/   # Videos of dogs sleeping (normal behavior)
│   ├── walking/    # Videos of dogs walking normally (normal behavior)
│   └── resting/    # Videos of dogs resting calmly (normal behavior)
├── val/            # 10% of your videos go here (same structure)
└── test/           # 10% of your videos go here (same structure)
```

## Instructions

1. **Organize your videos:**
   - **Distressed behaviors** → `pacing/` or `scratching/`
   - **Normal behaviors** → `sleeping/`, `walking/`, or `resting/`

2. **Split your videos:**
   - 80% → `train/` folders
   - 10% → `val/` folders
   - 10% → `test/` folders

3. **Supported video formats:**
   - `.mp4`
   - `.avi`
   - `.mov`
   - `.mkv`

4. **Video requirements:**
   - 10-60 seconds long
   - Clear view of the dog
   - Various lighting conditions and angles
   - Minimum 50-100 videos per behavior class recommended

## Example

```
train/
├── pacing/
│   ├── dog1_pacing.mp4
│   ├── dog2_pacing.mp4
│   └── ...
├── scratching/
│   ├── dog1_scratching.mp4
│   └── ...
└── ...
```

## Next Steps

After uploading videos:
1. Run: `python scripts/extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing`
2. Repeat for all behaviors and splits
3. Then run: `python scripts/prepare_yolo_dataset.py ...`
4. Finally: `python scripts/train_dog_behavior.py`

See `DOG_DISTRESS_TRAINING_GUIDE.md` for detailed instructions.

