# Train on Google Colab - Quick Guide

## Step-by-Step Instructions

### 1. Open Colab Notebook
- Open `Train_on_Colab.ipynb` in Google Colab
- Or create a new notebook and copy the cells

### 2. Enable GPU
- Runtime → Change runtime type → GPU → Save

### 3. Upload Videos

**Option A: Upload ZIP file**
1. Organize videos in folders on your computer:
   ```
   pig_videos/
   ├── train/
   │   ├── tail_biting/
   │   ├── ear_biting/
   │   ├── aggression/
   │   ├── eating/
   │   ├── sleeping/
   │   └── rooting/
   └── val/
       └── (same structure)
   ```
2. Zip the `pig_videos` folder
3. Upload ZIP in Colab
4. Unzip in the notebook

**Option B: Upload directly**
- Use Colab's file browser to upload videos to each folder
- Or use `files.upload()` in the notebook

### 4. Run All Cells
- Run each cell in order (Shift+Enter)
- Training will take 1-4 hours depending on data size

### 5. Download Model
- After training, download `best.pt` model file
- Save it to your project's `models/` folder

### 6. Use Model Locally
```bash
export YOLO_MODEL_PATH="models/best.pt"
cd backend && python app.py
```

---

## Tips

- **GPU**: Always use GPU runtime (free tier available)
- **Data Size**: Colab has ~15GB storage - compress videos if needed
- **Timeouts**: Colab sessions timeout after ~12 hours of inactivity
- **Download**: Download model immediately after training

---

## Quick Copy-Paste

If you prefer to create your own notebook, the key cells are in `Train_on_Colab.ipynb`.

---

**That's it! Train on Colab, download model, use locally!** 🐷

