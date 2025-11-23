# Upload Your Pig Videos Here

## Folder Structure

```
pig_training/
├── train/          ← 80% of videos
│   ├── tail_biting/    ← Distress
│   ├── ear_biting/     ← Distress
│   ├── aggression/     ← Distress
│   ├── eating/         ← Normal
│   ├── sleeping/       ← Normal
│   └── rooting/        ← Normal
├── val/            ← 10% of videos
└── test/           ← 10% of videos
```

## Instructions

1. Upload videos to appropriate folders
2. Split: 80% train, 10% val, 10% test
3. Formats: `.mp4`, `.avi`, `.mov`, `.mkv`
4. Length: 10-60 seconds
5. Minimum: 50-100 videos per class

## After Uploading

Run: `./scripts/run_all_steps.sh`
