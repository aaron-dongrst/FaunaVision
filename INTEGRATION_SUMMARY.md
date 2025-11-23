# YOLO Integration Summary

## Overview

The backend has been updated to integrate YOLO for behavior classification. The system now:

1. **Processes videos with YOLO** to classify behaviors frame-by-frame
2. **Calculates time percentages** for each behavior class (scratching, pacing, sleeping)
3. **Sends percentages to OpenAI/Gemini** along with animal parameters
4. **Returns health assessment** to the frontend

## Changes Made

### Backend (`backend/app.py`)

- ✅ Replaced `VisionEngine` with `YOLOBehaviorClassifier`
- ✅ Updated `process_video_with_model()` → `process_video_with_yolo()`
- ✅ Returns behavior percentages instead of single activity
- ✅ Updated OpenAI/Gemini integration to use percentages
- ✅ Added support for both OpenAI and Gemini (set `USE_GEMINI=true` to use Gemini)

### New Module (`src/yolo_behavior_classifier.py`)

- ✅ `YOLOBehaviorClassifier` class for video analysis
- ✅ Frame-by-frame processing with configurable interval
- ✅ Percentage calculation (always sums to 100%)
- ✅ Primary behavior detection

### Frontend (`frontend/src/components/AnalysisResults.js`)

- ✅ Updated to display behavior percentages
- ✅ Added visual percentage bars
- ✅ Shows primary behavior with percentage

### Requirements (`requirements.txt`)

- ✅ Added `ultralytics>=8.0.0` for YOLO support

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train YOLO Model

Follow the guide in `YOLO_TRAINING_GUIDE.md` to:
- Collect and annotate training videos
- Train YOLO classification model
- Save model as `.pt` file

### 3. Configure Environment Variables

```bash
# Required: Path to trained YOLO model
export YOLO_MODEL_PATH="models/behavior_classifier.pt"

# Required: OpenAI or Gemini API key
export OPENAI_API_KEY="your-openai-key"
# OR
export GEMINI_API_KEY="your-gemini-key"
export USE_GEMINI=true  # Set to use Gemini instead of OpenAI

# Optional
export PORT=5000
export FLASK_DEBUG=True
```

### 4. Run Backend

```bash
cd backend
python app.py
```

## API Response Format

The `/analyze` endpoint now returns:

```json
{
  "species": "bear",
  "behavior_percentages": {
    "scratching": 0.15,
    "pacing": 0.60,
    "sleeping": 0.25
  },
  "primary_behavior": "pacing",
  "primary_behavior_percentage": 0.60,
  "length_seconds": 120.5,
  "length_minutes": 2.01,
  "is_healthy": false,
  "reasoning": "The animal spent 60% of time pacing...",
  "recommendations": "Consider environmental enrichment..."
}
```

## Behavior Classes

Currently configured behaviors:
- `scratching`: Animal scratching itself
- `pacing`: Animal pacing back and forth
- `sleeping`: Animal sleeping/resting

To add more behaviors:
1. Update `behavior_classes` in `src/yolo_behavior_classifier.py`
2. Retrain YOLO model with new classes
3. Update frontend display if needed

## Next Steps

1. **Train YOLO Model**: Follow `YOLO_TRAINING_GUIDE.md`
2. **Test Integration**: Use test videos to verify percentages
3. **Fine-tune**: Adjust frame interval, confidence thresholds
4. **Deploy**: Set up production environment with trained model

## Troubleshooting

### YOLO Model Not Loading
- Check `YOLO_MODEL_PATH` environment variable
- Verify model file exists and is valid `.pt` file
- Check logs for initialization errors

### Percentages Don't Sum to 100%
- This should be handled automatically by normalization
- Check logs for processing errors

### Low Accuracy
- Retrain model with more diverse data
- Adjust confidence threshold
- Consider different frame intervals

