"""
Monkeypox Skin Lesion Detector
Streamlit web app — upload a skin image, get a prediction
Model: Custom CNN trained with TensorFlow/Keras
"""

import streamlit as st
import numpy as np
import os
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monkeypox Skin Lesion Detector",
    page_icon="🔬",
    layout="centered"
)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    model_path = "models/monkeypox_classifier.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ── Preprocess image to match training pipeline ───────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((64, 64))
    arr = np.array(img) / 255.0          # normalise to [0,1]
    return np.expand_dims(arr, axis=0)   # add batch dimension

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model):
    processed = preprocess(image)
    preds = model.predict(processed, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    return class_idx, confidence, preds

CLASS_NAMES = ["Monkeypox", "Others"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔬 Monkeypox Skin Lesion Detector")

st.markdown("""
A deep learning classifier trained on clinical skin lesion images to detect **Monkeypox** vs other skin conditions.

Upload a photo of a skin lesion to get a prediction.

---
""")

st.warning(
    "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
    "It is **not** a medical diagnosis. Always consult a qualified healthcare professional."
)

st.divider()

# Check model
model = load_model()
if model is None:
    st.error(
        "Model file not found at `models/monkeypox_classifier.h5`. "
        "Please run the training notebook first."
    )
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG, or PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analysing image..."):
            class_idx, confidence, all_probs = predict(image, model)

        predicted_label = CLASS_NAMES[class_idx]

        st.markdown("### 🧠 Result")
        if predicted_label == "Monkeypox":
            st.error(f"**{predicted_label} Detected**")
            st.markdown("This image shows features **consistent with monkeypox** lesions.")
        else:
            st.success(f"**{predicted_label}**")
            st.markdown("This image does **not** show clear signs of monkeypox.")

        st.markdown("### 📊 Confidence")
        st.progress(confidence)
        st.markdown(f"**{confidence*100:.1f}%** confidence")

        st.markdown("### 📈 Class Probabilities")
        for name, prob in zip(CLASS_NAMES, all_probs):
            st.markdown(f"**{name}:** {prob*100:.1f}%")
            st.progress(float(prob))

st.divider()
st.caption(
    "Built with TensorFlow/Keras + Streamlit  |  "
    "Dataset: [Soumya Thamke — Kaggle](https://www.kaggle.com/datasets/soumyathamke/monkeypoxvssmallpox)"
)
