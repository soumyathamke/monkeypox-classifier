# 🔬 Monkeypox Skin Lesion Classifier

A binary CNN classifier that detects **Monkeypox** from clinical skin lesion images, built with TensorFlow/Keras and deployed via a Streamlit web app.

---

## Overview

During the 2022 monkeypox outbreak, clinical image data was extremely scarce. This project tackles that challenge head-on — the dataset was **self-curated and published on Kaggle** by the project lead, and a custom CNN was trained using data augmentation to handle the limited sample size, which is standard practice in medical imaging ML.

---

## Dataset

- **Published by:** [Soumya Thamke — Kaggle](https://www.kaggle.com/datasets/soumyathamke/monkeypoxvssmallpox)
- **Classes:** Monkeypox, Others
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

## Model

- **Architecture:** Custom CNN — 4 Conv blocks + BatchNorm + Dropout
- **Framework:** TensorFlow / Keras
- **Input size:** 64 × 64 × 3
- **Output:** Softmax over 2 classes
- **Augmentation:** Horizontal flip, rotation, zoom, shear, brightness variation
- **Callbacks:** ModelCheckpoint (best val accuracy), EarlyStopping

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/monkeypox-classifier.git
cd monkeypox-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Download from Kaggle and place in data/ folder

# 4. Train
jupyter notebook monkeypox_classifier.ipynb
# or run cells top to bottom — model saves to models/monkeypox_classifier.h5

# 5. Launch app
streamlit run app.py
```

---

## Project Structure

```
monkeypox-classifier/
├── monkeypox_classifier.ipynb   # Training notebook (commented)
├── app.py                       # Streamlit web app
├── requirements.txt
├── models/                      # Saved model weights (after training)
└── data/                        # Dataset (not committed to Git)
```

---

## Team

Built by a team of 4 (2023). Dataset curated and published by Soumya Thamke.

---

## Disclaimer

For **research and educational purposes only**. Not intended for clinical diagnosis.
