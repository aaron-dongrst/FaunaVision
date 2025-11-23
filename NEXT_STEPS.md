# Next Steps After Uploading Files

## ✅ You've Uploaded:
- 8 video files → `data/videos/`
- 8 JSON annotation files → `data/annotations/`

## 🚀 Next: Run Training

### Step 1: Verify Files (Optional)
```bash
# Check files are there
ls data/videos/
ls data/annotations/
```

### Step 2: Run Training Script
```bash
./scripts/train_from_annotations.sh data/videos data/annotations
```

This will:
1. ✅ Parse all JSON files
2. ✅ Extract labeled pig crops from videos
3. ✅ Organize crops by behavior class
4. ✅ Split into train/val sets (80/20)
5. ✅ Prepare YOLO dataset
6. ✅ Train the model (takes 1-4 hours)

### Step 3: Wait for Training
- Training will take 1-4 hours depending on:
  - Number of pig crops extracted
  - Your GPU/CPU speed
  - Model size

### Step 4: Get Your Model
After training completes, your model will be at:
```
pig_behavior_classification/yolov8_pig_behavior/weights/best.pt
```

### Step 5: Use the Model
```bash
# Set model path
export YOLO_MODEL_PATH="pig_behavior_classification/yolov8_pig_behavior/weights/best.pt"
export OPENAI_API_KEY="your-key-here"

# Start backend
cd backend && python app.py

# Start frontend (new terminal)
cd frontend && npm start
```

---

## ⚠️ Troubleshooting

**"No matching JSON found"**
- Check JSON filenames match video filenames exactly (except extension)
- Example: `video1.mp4` ↔ `video1.json`

**"Unknown behavior label"**
- Check behavior labels in JSON are exactly: tail_biting, ear_biting, aggression, eating, sleeping, or rooting

**Training takes too long**
- Reduce epochs in `scripts/train_pig_behavior.py` (change `EPOCHS = 50` instead of 100)
- Use smaller model (change `MODEL_SIZE = "n"` to "n" for nano)

---

**Ready? Run the training script now!** 🐷

