# Exact Steps: Dog Distress Detection Setup

## What You Need to Do (Step-by-Step)

### Step 1: Install Dependencies (One Time)

```bash
# Make sure you're in the project root
cd /Users/aarondong/Desktop/Faunavision

# Install Python dependencies
pip install -r requirements.txt

# Install YOLO
pip install ultralytics
```

---

### Step 2: Upload Your Dog Videos

Upload your dog behavior videos to these folders:

**For Training (80% of videos):**
- `data/dog_training/train/pacing/` ← Dogs pacing back and forth
- `data/dog_training/train/scratching/` ← Dogs scratching excessively
- `data/dog_training/train/sleeping/` ← Dogs sleeping
- `data/dog_training/train/walking/` ← Dogs walking normally
- `data/dog_training/train/resting/` ← Dogs resting calmly

**For Validation (10% of videos):**
- `data/dog_training/val/pacing/`
- `data/dog_training/val/scratching/`
- `data/dog_training/val/sleeping/`
- `data/dog_training/val/walking/`
- `data/dog_training/val/resting/`

**For Testing (10% of videos):**
- `data/dog_training/test/pacing/`
- `data/dog_training/test/scratching/`
- `data/dog_training/test/sleeping/`
- `data/dog_training/test/walking/`
- `data/dog_training/test/resting/`

**Requirements:**
- Minimum 50-100 videos per behavior class (more is better!)
- Video formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Video length: 10-60 seconds

---

### Step 3: Run Training Pipeline

Once videos are uploaded, run:

```bash
# Make script executable (one time)
chmod +x scripts/run_all_steps.sh

# Run the complete training pipeline
./scripts/run_all_steps.sh
```

**This will:**
1. Extract frames from all videos (takes 10-30 minutes)
2. Prepare YOLO dataset (takes 5-10 minutes)
3. Train the model (takes 1-4 hours depending on GPU)

**OR run steps manually:**

```bash
# Step 3a: Extract frames (repeat for each behavior)
python scripts/extract_frames.py data/dog_training/train/pacing data/dog_frames/train/pacing
python scripts/extract_frames.py data/dog_training/train/scratching data/dog_frames/train/scratching
python scripts/extract_frames.py data/dog_training/train/sleeping data/dog_frames/train/sleeping
python scripts/extract_frames.py data/dog_training/train/walking data/dog_frames/train/walking
python scripts/extract_frames.py data/dog_training/train/resting data/dog_frames/train/resting

# Repeat for val/ and test/...

# Step 3b: Prepare dataset
python scripts/prepare_yolo_dataset.py data/dog_frames/train data/yolo_dataset/train
python scripts/prepare_yolo_dataset.py data/dog_frames/val data/yolo_dataset/val

# Step 3c: Train model
python scripts/train_dog_behavior.py
```

---

### Step 4: Test Your Model

After training completes, test it:

```bash
python scripts/test_dog_model.py data/test_videos/your_dog_video.mp4 dog_behavior_classification/yolov8_dog_behavior/weights/best.pt
```

---

### Step 5: Configure Backend

```bash
# Set the path to your trained model
export YOLO_MODEL_PATH="dog_behavior_classification/yolov8_dog_behavior/weights/best.pt"

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# OR use Gemini instead
export GEMINI_API_KEY="your-gemini-api-key-here"
export USE_GEMINI=true
```

---

### Step 6: Start Backend Server

```bash
cd backend
python app.py
```

The backend will start on `http://localhost:5000`

---

### Step 7: Start Frontend

In a new terminal:

```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000`

---

### Step 8: Use the Application

1. Open browser to `http://localhost:3000`
2. Click "Add New Dog"
3. Enter dog information:
   - Species: "dog" (or specific breed)
   - Age: e.g., "3 years"
   - Diet: e.g., "commercial dog food"
   - Health Conditions: e.g., "none" or any existing conditions
4. Upload a dog video
5. Click "Analyze Dog Behavior"
6. View results:
   - Behavior percentages (pacing, scratching, sleeping, etc.)
   - Distress assessment
   - Health recommendations

---

## Summary Checklist

- [ ] Install dependencies (`pip install -r requirements.txt` and `pip install ultralytics`)
- [ ] Upload videos to `data/dog_training/` folders
- [ ] Run training: `./scripts/run_all_steps.sh`
- [ ] Wait for training to complete (1-4 hours)
- [ ] Test model: `python scripts/test_dog_model.py ...`
- [ ] Set environment variables (`YOLO_MODEL_PATH`, `OPENAI_API_KEY`)
- [ ] Start backend: `cd backend && python app.py`
- [ ] Start frontend: `cd frontend && npm start`
- [ ] Use the app in browser!

---

## Troubleshooting

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"Dataset not found"**
- Make sure you ran frame extraction first
- Check that `data/yolo_dataset/data.yaml` exists

**"Model not loading"**
- Check `YOLO_MODEL_PATH` is set correctly
- Verify model file exists at the path

**Backend won't start**
- Check `OPENAI_API_KEY` or `GEMINI_API_KEY` is set
- Make sure port 5000 is not in use

**Frontend won't start**
- Make sure you ran `npm install` in frontend directory
- Check port 3000 is not in use

---

That's it! Follow these steps in order and you'll have a working dog distress detection system.

