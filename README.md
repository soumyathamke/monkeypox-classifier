# 🔬 Monkeypox Skin Lesion Classifier

A binary CNN classifier that detects **Monkeypox** from clinical skin lesion images, built with TensorFlow/Keras.

---

## Overview

During the 2022 monkeypox outbreak, clinical image data was extremely scarce and hard to source. This project tackles that directly — the dataset was **self-curated and published on Kaggle** by the project lead, and a custom CNN was trained using data augmentation to handle the limited sample size. This is the standard approach used in real-world medical imaging research.

---

## Demo

<img width="516" height="504" alt="image" src="https://github.com/user-attachments/assets/887a0ff5-0075-476d-97ee-991ca660df80" />


> Feed a skin lesion image → model returns class prediction with confidence score

---

## Dataset

- **Published by:** [Soumya Thamke — Kaggle](https://www.kaggle.com/datasets/soumyathamke/monkeypoxvssmallpox)
- **Classes:** Monkeypox, Others
- **Train:** ~2,100 augmented images | **Test:** 45 images
- **Structure:**
  ```
  data/
  ├── Train/
  │   ├── Monkeypox/
  │   └── Others/
  └── Test/
      ├── Monkeypox/
      └── Others/
  ```

---

## Model Architecture

Custom CNN built from scratch in Keras — 4 convolutional blocks with BatchNormalization and Dropout for regularisation.

```
Input (64×64×3)
→ Conv2D(32) + BatchNorm + MaxPool
→ Conv2D(64) + BatchNorm + MaxPool
→ Conv2D(128) + BatchNorm + MaxPool
→ Conv2D(256) + BatchNorm + MaxPool
→ Dropout(0.5) → Flatten
→ Dense(128) → Dropout(0.1)
→ Dense(256) → Dropout(0.25)
→ Dense(2, softmax)
```

- **Framework:** TensorFlow / Keras
- **Optimiser:** Adam | **Loss:** Categorical Crossentropy
- **Augmentation:** Horizontal flip, rotation (15°), zoom, shear, brightness variation
- **Callbacks:** ModelCheckpoint (saves best val accuracy), EarlyStopping (patience=7)
- **Best Validation Accuracy: 66.7%** on 45-image held-out test set

> Note: Validation variance is expected given the small test set — a known constraint in rare disease imaging.

---

## Results

| Metric | Value |
|---|---|
| Best Val Accuracy | 66.7% |
| Train Accuracy (final) | ~78% |
| Test Set Size | 45 images |
| Stopped at Epoch | 10 (early stopping) |

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/soumyathamke/monkeypox-classifier.git
cd monkeypox-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Download from Kaggle and place in data/ folder

# 4. Train — open and run all cells
jupyter notebook monkeypox_classifier.ipynb
# Best model saves to models/monkeypox_classifier.h5
```

---

## Project Structure

```
monkeypox-classifier/
├── monkeypox_classifier.ipynb   # Training notebook (fully commented)
├── requirements.txt
├── assets/
│   └── demo.png                 # Sample prediction screenshot
├── models/                      # Saved model weights (after training)
└── data/                        # Dataset (not committed to Git)
```

---

## Team

Built by a team of 4 (2023). Dataset curated and published by Soumya Thamke.

---

## Disclaimer

For **research and educational purposes only**. Not intended for clinical diagnosis.
