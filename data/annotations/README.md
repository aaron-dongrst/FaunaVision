# Annotation Format

## Expected JSON Structure

Each JSON file should correspond to a video file and contain a list of pig objects:

```json
[
  {
    "tracking_id": 1,
    "frames": [0, 1, 2, 3, ...],
    "bounding_box": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
    "behavior_label": "tail_biting",
    "visibility": 1.0,
    "ground_truth": true
  },
  {
    "tracking_id": 2,
    "frames": [10, 11, 12, ...],
    "bounding_box": [[x1, y1, x2, y2], ...],
    "behavior_label": "eating",
    "visibility": 0.9,
    "ground_truth": true
  }
]
```

## Fields

- **tracking_id**: Unique ID for each pig in the video
- **frames**: List of frame numbers where this pig appears
- **bounding_box**: List of bounding boxes, one per frame (format: [x1, y1, x2, y2] or [x, y, width, height])
- **behavior_label**: Behavior class (tail_biting, ear_biting, aggression, eating, sleeping, rooting)
- **visibility**: Visibility flag (0.0 to 1.0, typically 1.0 for visible)
- **ground_truth**: Boolean indicating if this is ground truth annotation

## Usage

1. Place your videos in `data/videos/`
2. Place corresponding JSON files in `data/annotations/` (same filename as video, but .json)
3. Run: `./scripts/train_from_annotations.sh data/videos data/annotations`

