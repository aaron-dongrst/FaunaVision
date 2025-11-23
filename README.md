# 🐷 PigVision - Pig Distress Detection

AI-powered system to detect pig distress behaviors through video analysis.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install ultralytics
cd frontend && npm install
```

### 2. Upload Videos and Annotations ⭐

**Option A: Using JSON Annotations (Recommended)**
1. Place videos in `data/videos/`
2. Place JSON annotation files in `data/annotations/` (same filename as video, but .json)
3. Run: `./scripts/train_from_annotations.sh data/videos data/annotations`

**Option B: Manual Organization**
Upload videos to `data/pig_training/train/` folders (tail_biting, ear_biting, etc.)
Then run: `./scripts/run_all_steps.sh`

### 4. Run Application
```bash
# Set environment variables
export YOLO_MODEL_PATH="pig_behavior_classification/yolov8_pig_behavior/weights/best.pt"
export OPENAI_API_KEY="your-key-here"

# Start backend
cd backend && python app.py

# Start frontend (new terminal)
cd frontend && npm start
```

Open `http://localhost:3000`

---

## Project Structure

```
Faunavision/
├── README.md              ← You are here
├── data/
│   └── pig_training/      ← ⭐ Upload videos here
├── scripts/               ← Training scripts
│   └── run_all_steps.sh  ← Run everything
├── frontend/              ← React UI
├── backend/               ← Flask API
└── src/                   ← Core modules
```

---

## Behavior Classes

**Distress (3):** tail_biting, ear_biting, aggression  
**Normal (3):** eating, sleeping, rooting

---

## Scripts

- `scripts/run_all_steps.sh` - Complete training pipeline
- `scripts/train_pig_behavior.py` - Train model
- `scripts/test_pig_model.py` - Test model

---

**Ready? Upload videos to `data/pig_training/` and run `./scripts/run_all_steps.sh`!**
