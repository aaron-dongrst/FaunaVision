# Training Scripts

## Quick Start

### Using JSON Annotations (Recommended)
```bash
./scripts/train_from_annotations.sh data/videos data/annotations
```

### Manual Video Organization
```bash
./scripts/run_all_steps.sh
```

## Scripts

1. **`parse_annotations.py`** - Parse JSON and extract labeled pig crops
2. **`prepare_yolo_from_crops.py`** - Prepare YOLO dataset from crops
3. **`train_from_annotations.sh`** - Complete pipeline using annotations
4. **`run_all_steps.sh`** - Pipeline for manually organized videos
5. **`train_pig_behavior.py`** - Train YOLO model
6. **`test_pig_model.py`** - Test trained model

## JSON Annotation Format

See `data/annotations/README.md` for expected JSON structure.
