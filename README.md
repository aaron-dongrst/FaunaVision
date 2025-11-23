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

### 2. Upload Videos ⭐
Upload pig behavior videos to `data/pig_training/`:

- **Distress behaviors:**
  - `train/tail_biting/` - Pigs biting tails
  - `train/ear_biting/` - Pigs biting ears
  - `train/aggression/` - Aggressive behavior

- **Normal behaviors:**
  - `train/eating/` - Pigs eating
  - `train/sleeping/` - Pigs sleeping
  - `train/rooting/` - Pigs rooting

**Split:** 80% train, 10% val, 10% test

### 3. Train Model
```bash
./scripts/run_all_steps.sh
```

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
