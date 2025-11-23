# 🐕 DogVision - Dog Distress Detection System

AI-powered system to detect dog distress behaviors through video analysis using YOLO and OpenAI/Gemini.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install ultralytics
cd frontend && npm install
```

### 2. Upload Dog Videos ⭐
Upload your dog behavior videos to:
- `data/dog_training/train/pacing/` - Dogs pacing (distress)
- `data/dog_training/train/scratching/` - Dogs scratching (distress)
- `data/dog_training/train/sleeping/` - Dogs sleeping (normal)
- `data/dog_training/train/walking/` - Dogs walking (normal)
- `data/dog_training/train/resting/` - Dogs resting (normal)

**Split your videos:**
- 80% → `train/` folders
- 10% → `val/` folders
- 10% → `test/` folders

### 3. Train Model
```bash
./scripts/run_all_steps.sh
```

### 4. Configure & Run
```bash
# Set environment variables
export YOLO_MODEL_PATH="dog_behavior_classification/yolov8_dog_behavior/weights/best.pt"
export OPENAI_API_KEY="your-key-here"

# Start backend
cd backend && python app.py

# Start frontend (in new terminal)
cd frontend && npm start
```

### 5. Use the App
Open `http://localhost:3000` in your browser!

---

## 📁 Project Structure

```
Faunavision/
├── README.md                 ← You are here
├── docs/                     ← All documentation
│   ├── START_HERE.md        ← Quick start guide
│   ├── EXACT_STEPS.md       ← Step-by-step instructions
│   └── ...
├── data/
│   └── dog_training/        ← ⭐ UPLOAD VIDEOS HERE
│       ├── train/           ← 80% of videos
│       ├── val/             ← 10% of videos
│       └── test/            ← 10% of videos
├── scripts/                 ← Training scripts
│   └── run_all_steps.sh    ← Run everything
├── frontend/                ← React UI (DogVision)
├── backend/                 ← Flask API
└── src/                     ← Core modules
```

---

## 📚 Documentation

All guides are in the `docs/` folder:

- **`docs/START_HERE.md`** - Quick start guide
- **`docs/EXACT_STEPS.md`** - Complete step-by-step instructions
- **`docs/DOG_DISTRESS_TRAINING_GUIDE.md`** - Detailed training guide
- **`docs/YOLO_TRAINING_GUIDE.md`** - General YOLO guide
- **`docs/INTEGRATION_SUMMARY.md`** - Technical integration details

---

## 🎯 What You Need to Do

1. ✅ **Install dependencies** (one time)
2. ⭐ **Upload dog videos** to `data/dog_training/` folders
3. ✅ **Run training**: `./scripts/run_all_steps.sh`
4. ✅ **Configure & run** backend and frontend
5. ✅ **Use the app**!

---

## 🐕 How It Works

1. **Upload video** → Frontend
2. **Process with YOLO** → Classify behaviors (pacing, scratching, sleeping, etc.)
3. **Calculate percentages** → Time spent in each behavior
4. **Analyze with AI** → Determine distress level
5. **Display results** → Behavior breakdown + health recommendations

---

## 🎨 Features

- ✅ Dog-themed UI (warm brown/tan colors)
- ✅ Video upload and analysis
- ✅ Behavior classification (pacing, scratching, sleeping, walking, resting)
- ✅ Distress detection based on behavior percentages
- ✅ Health recommendations via OpenAI/Gemini

---

## 📋 Requirements

- Python 3.8+
- Node.js 14+
- Dog behavior videos (50-100+ per behavior class)
- OpenAI API key (or Gemini API key)

---

## 🔧 Scripts

- `scripts/run_all_steps.sh` - Run complete training pipeline
- `scripts/extract_frames.py` - Extract frames from videos
- `scripts/prepare_yolo_dataset.py` - Prepare YOLO dataset
- `scripts/train_dog_behavior.py` - Train YOLO model
- `scripts/test_dog_model.py` - Test trained model

---

## 📖 See Also

- `docs/START_HERE.md` - Quick start
- `docs/EXACT_STEPS.md` - Detailed instructions
- `backend/README.md` - Backend API docs
- `frontend/README.md` - Frontend docs

---

## 🚀 Ready to Start?

1. Read `docs/START_HERE.md`
2. Upload videos to `data/dog_training/`
3. Run `./scripts/run_all_steps.sh`
4. Start using DogVision!

---

**Made with ❤️ for dog health monitoring**
