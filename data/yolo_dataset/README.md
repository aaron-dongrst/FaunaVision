# YOLO Dataset Directory

This directory will be automatically populated after running the dataset preparation scripts.

## Structure (created automatically):
```
yolo_dataset/
├── train/
│   ├── images/  # All training images
│   └── labels/  # Corresponding labels (.txt files)
├── val/
│   ├── images/  # All validation images
│   └── labels/  # Corresponding labels (.txt files)
└── data.yaml    # Dataset configuration
```

## How it's created:

1. Videos are extracted to frames → `data/dog_frames/`
2. Frames are prepared for YOLO → This directory
3. Training uses this directory

## Don't manually edit files here!

The scripts will populate this directory automatically.

