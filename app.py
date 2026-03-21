"""
=============================================================
app.py — Streamlit Web Application
=============================================================
Chest Cancer Detection using CNN
=============================================================

Run with:
    streamlit run app.py

Features:
  - Upload CT scan image (JPG, PNG, BMP)
  - Real-time CNN inference
  - Confidence bar chart
  - Medical disclaimer
=============================================================
"""

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title  = "Chest Cancer Detection | CNN",
    page_icon   = "🫁",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Constants ──────────────────────────────────────────────
MODEL_PATH   = Path("models/best_model.h5")
MAPPING_PATH = Path("data/processed/class_mapping.json")
IMAGE_SIZE   = (224, 224)

CLASS_INFO = {
    "adenocarcinoma": {
        "label": "Adenocarcinoma",
        "color": "#FF6B6B",
        "icon":  "🔴",
        "desc":  "Most common type of non-small cell lung cancer. Originates in the outer regions of the lung.",
    },
    "large.cell.carcinoma": {
        "label": "Large Cell Carcinoma",
        "color": "#FFA500",
        "icon":  "🟠",
        "desc":  "Fast-growing cancer that can appear in any part of the lung.",
    },
    "squamous.cell.carcinoma": {
        "label": "Squamous Cell Carcinoma",
        "color": "#FFD700",
        "icon":  "🟡",
        "desc":  "Usually found in the central part of the lung near a main airway (bronchus).",
    },
    "normal": {
        "label": "Normal",
        "color": "#4CAF50",
        "icon":  "🟢",
        "desc":  "No signs of cancerous tissue detected in the CT scan.",
    },
}


# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Keras model. Cached across sessions."""
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def load_class_mapping():
    """Load index → class name mapping."""
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH) as f:
            mapping = json.load(f)
        return {int(k): v for k, v in mapping.items()}
    # Fallback mapping
    return {
        0: "adenocarcinoma",
        1: "large.cell.carcinoma",
        2: "normal",
        3: "squamous.cell.carcinoma",
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for model input:
      1. Convert to RGB
      2. Resize to 224×224
      3. Normalize [0, 255] → [0, 1]
      4. Add batch dimension
    """
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # Shape: (1, 224, 224, 3)


def predict(model, image: Image.Image, class_mapping: dict) -> dict:
    """
    Run inference and return:
      - predicted class name
      - confidence score
      - all class probabilities
    """
    inp          = preprocess_image(image)
    probs        = model.predict(inp, verbose=0)[0]  # Shape: (4,)
    pred_idx     = int(np.argmax(probs))
    pred_class   = class_mapping.get(pred_idx, "unknown")
    confidence   = float(probs[pred_idx])

    class_probs = {
        class_mapping.get(i, f"class_{i}"): float(p)
        for i, p in enumerate(probs)
    }

    return {
        "predicted_class": pred_class,
        "confidence":      confidence,
        "probabilities":   class_probs,
        "pred_idx":        pred_idx,
    }


# ── Sidebar ────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-lung-human-organs-flaticons-lineal-color-flat-icons-2.png", width=80)
        st.title("🫁 Cancer Detector")
        st.markdown("---")

        st.markdown("### About")
        st.info(
            "This app uses a Convolutional Neural Network (CNN) trained on "
            "chest CT-scan images to classify 4 types of lung conditions."
        )

        st.markdown("### Classes")
        for key, info in CLASS_INFO.items():
            st.markdown(f"{info['icon']} **{info['label']}**")

        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(
            "| Item | Value |\n"
            "|------|-------|\n"
            "| Architecture | Custom CNN + VGG16 |\n"
            "| Input Size | 224 × 224 |\n"
            "| Classes | 4 |\n"
            "| Framework | TensorFlow/Keras |"
        )

        st.markdown("---")
        st.warning(
            "⚠️ **Medical Disclaimer**\n\n"
            "This tool is for educational and research purposes only. "
            "It is NOT a substitute for professional medical diagnosis. "
            "Always consult a qualified physician."
        )


# ── Main App ───────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────
    st.title("🫁 Chest Cancer Detection using CNN")
    st.markdown(
        "Upload a **Chest CT-scan image** and the AI model will classify it into "
        "one of four categories: **Adenocarcinoma**, **Large Cell Carcinoma**, "
        "**Squamous Cell Carcinoma**, or **Normal**."
    )
    st.markdown("---")

    # ── Load model ────────────────────────────────────────
    model         = load_model()
    class_mapping = load_class_mapping()

    if model is None:
        st.error(
            "⚠️ **Model not found.**\n\n"
            "Please train the model first:\n"
            "```bash\n"
            "python src/train_cnn.py\n"
            "```\n"
            "Or run the full pipeline:\n"
            "```bash\n"
            "dvc repro\n"
            "```"
        )
        st.info("The app UI is fully functional. Train the model to enable predictions.")

    # ── File uploader ────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload CT Scan Image")
        uploaded_file = st.file_uploader(
            label  = "Choose a CT scan image",
            type   = ["jpg", "jpeg", "png", "bmp"],
            help   = "Upload a chest CT-scan image for classification."
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            st.markdown(f"**File:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} px")
            st.markdown(f"**Mode:** {image.mode}")

            st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    # ── Prediction ───────────────────────────────────────
    with col2:
        st.markdown("### 🔬 Prediction Results")

        if uploaded_file is None:
            st.info("👈 Please upload a CT scan image to get a prediction.")
            return

        if model is None:
            st.warning("Model not loaded. Please train the model first.")
            return

        with st.spinner("🔄 Analyzing CT scan..."):
            image   = Image.open(uploaded_file)
            result  = predict(model, image, class_mapping)

        pred_class  = result["predicted_class"]
        confidence  = result["confidence"]
        probs       = result["probabilities"]

        cls_info = CLASS_INFO.get(pred_class, {
            "label": pred_class, "color": "#888888", "icon": "⚪", "desc": ""
        })

        # ── Main prediction banner ────────────────────────
        if pred_class == "normal":
            st.success(
                f"{cls_info['icon']} **Prediction: {cls_info['label']}**\n\n"
                f"Confidence: **{confidence:.2%}**"
            )
        else:
            st.error(
                f"{cls_info['icon']} **Prediction: {cls_info['label']}**\n\n"
                f"Confidence: **{confidence:.2%}**"
            )

        # ── Class description ─────────────────────────────
        st.markdown(f"**About this condition:** {cls_info['desc']}")
        st.markdown("---")

        # ── Confidence bars ───────────────────────────────
        st.markdown("#### 📊 Confidence Scores")

        for cls_key, prob in sorted(probs.items(), key=lambda x: -x[1]):
            info  = CLASS_INFO.get(cls_key, {"label": cls_key, "color": "#888", "icon": "⚫"})
            label = info["label"]
            pct   = prob * 100

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(
                    min(prob, 1.0),
                    text=f"{info['icon']} {label}"
                )
            with col_b:
                st.markdown(f"**{pct:.1f}%**")

        # ── Confidence chart ──────────────────────────────
        st.markdown("#### 📈 Probability Distribution")

        import plotly.graph_objects as go

        labels = [CLASS_INFO.get(k, {"label": k})["label"] for k in probs.keys()]
        values = list(probs.values())
        colors = [CLASS_INFO.get(k, {"color": "#888"})["color"] for k in probs.keys()]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Probability",
            yaxis_range=[0, 1.1],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sample images hint ────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ How to get test images?"):
        st.markdown("""
        **Option 1:** Download from Kaggle:
        ```bash
        kaggle datasets download -d mohamedhanyyy/chest-ctscan-images
        unzip chest-ctscan-images.zip -d data/raw/
        ```
        Then use images from `data/raw/test/` for testing.

        **Option 2:** Use any grayscale or color chest CT scan image.

        **Supported formats:** JPG, JPEG, PNG, BMP
        """)

    with st.expander("🛠️ MLflow Experiments"):
        st.markdown("""
        To view all training experiments in MLflow:
        ```bash
        mlflow ui --backend-store-uri mlruns
        ```
        Then open: http://localhost:5000
        """)


if __name__ == "__main__":
    main()
