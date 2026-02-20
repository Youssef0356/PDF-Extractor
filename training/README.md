# Tesseract Fine-Tuning for Equipment Labels

## Overview

This directory contains scripts to fine-tune Tesseract's LSTM engine on your
specific equipment label images. The more data you collect, the better the OCR
accuracy becomes.

## Quick Start

### Step 1: Collect Training Data

As you process labels with `ImprovedOCR.py`, training pairs are automatically
saved to `training/data/`. You can also manually add training data:

```bash
# From an image (shows OCR output for you to correct)
python training/collect_training_data.py Rosemount.jpeg

# With corrected ground truth
python training/collect_training_data.py Rosemount.jpeg "ROSEMOUNT TEMPERATURE TRANSMITTER MODEL 3144..."

# From a PDF
python training/collect_training_data.py document.pdf

# Check your training data status
python training/collect_training_data.py --stats
```

### Step 2: Fine-Tune Tesseract

Once you have ≥10 training pairs with correct ground truth:

```bash
python training/fine_tune_tesseract.py
```

This will:
1. ✅ Check prerequisites (Tesseract 5.x, training tools, base model)
2. ✅ Generate LSTMF training files
3. ✅ Extract LSTM from the base English model
4. ✅ Fine-tune for 400 iterations on your data
5. ✅ Create `equipment.traineddata` and install it

### Step 3: Use the Custom Model

`ImprovedOCR.py` will **automatically detect** and use the custom model.
No code changes needed.

## Prerequisites

- **Tesseract 5.x** with training tools (combine_tessdata, lstmtraining)
- **Base model**: `eng.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
- **Training data**: ≥10 image-text pairs (more = better)

## Directory Structure

```
training/
├── README.md                     # This file
├── collect_training_data.py      # Collect image-text pairs
├── fine_tune_tesseract.py        # Run fine-tuning
├── data/                         # Training pairs (auto-created)
│   ├── equipment_0001.tif        # Image
│   ├── equipment_0001.gt.txt     # Ground truth text
│   └── ...
└── output/                       # Training output (auto-created)
    ├── lstmf/                    # Training files
    ├── lstm/                     # Extracted base model
    ├── checkpoints/              # Training checkpoints
    └── equipment.traineddata     # Final custom model
```

## Tips for Better Results

1. **More diverse data** → better generalization
2. **Accurate ground truth** → essential (wrong text = wrong model)
3. **Include hard cases** → images that OCR currently struggles with
4. **Mix scales/quality** → helps the model handle varied input
5. **Iterate** → collect more data, re-train, repeat
