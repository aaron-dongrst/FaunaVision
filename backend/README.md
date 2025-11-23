# Backend API

Flask API for pig behavior analysis.

## Setup

```bash
export YOLO_MODEL_PATH="pig_behavior_classification/yolov8_pig_behavior/weights/best.pt"
export OPENAI_API_KEY="your-key-here"
python app.py
```

## Endpoints

- `GET /health` - Health check
- `POST /analyze` - Analyze pig video

## Request Format

```bash
curl -X POST http://localhost:5000/analyze \
  -F "video=@video.mp4" \
  -F "species=pig" \
  -F "age=6 months" \
  -F "diet=commercial feed"
```

## Response

Returns behavior percentages and health assessment.
