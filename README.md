
#  Chest Cancer Classification using CNN and MLOps

> **A production-grade Deep Learning + MLOps project for classifying chest CT-scan images into 4 cancer categories using a custom CNN and VGG16 transfer learning — tracked with MLflow and versioned with DVC.**

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [CNN Architecture](#cnn-architecture)
- [MLflow Experiments](#mlflow-experiments)
- [DVC Pipeline](#dvc-pipeline)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Streamlit App](#streamlit-app)
- [Docker Instructions](#docker-instructions)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds an **end-to-end machine learning pipeline** for detecting chest cancer from CT-scan images. It demonstrates:

- Custom CNN and VGG16 transfer learning
- 🔬 8 MLflow experiments with hyperparameter tracking
-  DVC pipeline for reproducibility
-  Streamlit web app for real-time predictions
-  Docker containerization
-  Full evaluation suite (confusion matrix, classification report, F1)

---

## 📂 Dataset

| Item | Detail |
|------|--------|
| Source | [Kaggle: Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| Modality | CT Scan (grayscale/color) |
| Classes | 4 (see below) |
| Splits | Train / Validation / Test |

### Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | **Adenocarcinoma** | Most common lung cancer, starts in outer regions |
| 2 | **Large Cell Carcinoma** | Fast-growing, found anywhere in lung |
| 3 | **Squamous Cell Carcinoma** | Near central airways |
| 4 | **Normal** | No cancerous tissue |

### Download Dataset

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d mohamedhanyyy/chest-ctscan-images
unzip chest-ctscan-images.zip -d data/raw/

# Option 2: Manual download from Kaggle
# → Place in data/raw/train/, data/raw/valid/, data/raw/test/
```

---

## 🗂️ Project Structure

```
CancerDetectionMLOps/
│
├── data/
│   ├── raw/                         # Dataset (tracked by DVC)
│   │   ├── train/
│   │   │   ├── adenocarcinoma/
│   │   │   ├── large.cell.carcinoma/
│   │   │   ├── squamous.cell.carcinoma/
│   │   │   └── normal/
│   │   ├── valid/
│   │   └── test/
│   └── processed/                   # Outputs: metrics, plots
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb      # EDA and dataset statistics
│   ├── 02_preprocessing.ipynb       # Augmentation and normalization
│   ├── 03_cnn_training.ipynb        # 8 MLflow experiments
│   └── 04_evaluation.ipynb          # Confusion matrix, F1, curves
│
├── src/
│   ├── data_ingestion.py            # Validate and count images
│   ├── preprocessing.py             # Build data generators
│   ├── train_cnn.py                 # Train with MLflow tracking
│   └── evaluate.py                  # Evaluate best model
│
├── models/                          # Saved models (.h5)
├── mlruns/                          # MLflow experiment logs
│
├── app.py                           # Streamlit web application
├── Dockerfile                       # Docker container config
├── dvc.yaml                         # DVC pipeline definition
├── params.yaml                      # Central hyperparameters
├── requirements.txt                 # Python dependencies
├── setup_conda_env.sh               # One-click conda setup
└── README.md
```

---

##  CNN Architecture

### Custom CNN

```
Input (224, 224, 3)
        ↓
Conv2D(32, 3×3, ReLU)  → BatchNorm → MaxPool(2×2) → Dropout(0.25)
        ↓
Conv2D(64, 3×3, ReLU)  → BatchNorm → MaxPool(2×2) → Dropout(0.25)
        ↓
Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
        ↓
Conv2D(256, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
        ↓
Flatten
        ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.5)
        ↓
Dense(4, Softmax)
```

### VGG16 Transfer Learning

```
Input (224, 224, 3)
        ↓
VGG16 (ImageNet weights, top removed, last layers fine-tuned)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.5)
        ↓
Dense(128, ReLU) → Dropout(0.3)
        ↓
Dense(4, Softmax)
```

---

## 📊 MLflow Experiments

8 experiments tracked with varying hyperparameters:

| Run | Model | Optimizer | LR | Dropout | Dense | Notes |
|-----|-------|-----------|-----|---------|-------|-------|
| Run 1 | CNN | Adam | 0.001 | 0.5 | 128 | Baseline |
| Run 2 | CNN | Adam | 0.0001 | 0.5 | 128 | Low LR |
| Run 3 | CNN | Adam | 0.001 | 0.6 | 128 | High Dropout |
| Run 4 | CNN | SGD | 0.01 | 0.5 | 128 | SGD Optimizer |
| Run 5 | CNN | Adam | 0.001 | 0.4 | 256 | Large Dense |
| Run 6 | CNN | RMSprop | 0.0005 | 0.5 | 128 | RMSprop |
| Run 7 | VGG16 | Adam | 0.0001 | — | — | Transfer Learning |
| Run 8 | VGG16 | Adam | 0.00005 | — | — | Aggressive Fine-Tune |

### View MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

---

## 🔁 DVC Pipeline

```
data/raw  ──→  data_ingestion  ──→  preprocessing  ──→  training  ──→  evaluation
                     ↓                    ↓                 ↓               ↓
               class_counts.json    done.flag        best_model.h5   test_metrics.json
```

### DVC Commands

```bash
# Initialize DVC (first time)
dvc init

# Track raw data
dvc add data/raw

# Run full pipeline
dvc repro

# View pipeline DAG
dvc dag

# Push data to remote (e.g., S3)
dvc remote add -d myremote s3://my-bucket/dvc
dvc push
```

---

##  Setup & Installation

### Conda Environment (Recommended)

```bash
# One-click setup
chmod +x setup_conda_env.sh
./setup_conda_env.sh

# Or manual:
conda create -n cancer_detection python=3.9 -y
conda activate cancer_detection
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name cancer_detection \
  --display-name "Python (cancer_detection)"
```

---

##  How to Run

### 1. Download Dataset

```bash
kaggle datasets download -d mohamedhanyyy/chest-ctscan-images
unzip chest-ctscan-images.zip -d data/raw/
```

### 2. Activate Environment

```bash
conda activate cancer_detection
```

### 3. Option A: Run Individual Scripts

```bash
# Stage 1: Ingest data
python src/data_ingestion.py

# Stage 2: Preprocess
python src/preprocessing.py

# Stage 3: Train (8 experiments with MLflow)
python src/train_cnn.py

# Stage 4: Evaluate best model
python src/evaluate.py
```

### 4. Option B: Full DVC Pipeline

```bash
dvc repro
```

### 5. Option C: Jupyter Notebooks

```bash
jupyter notebook
# Run notebooks in order: 01 → 02 → 03 → 04
```

---

##  Streamlit App

```bash
conda activate cancer_detection
streamlit run app.py
# Open: http://localhost:8501
```

**Features:**
- Upload CT scan image (JPG/PNG/BMP)
- Real-time CNN prediction
- Confidence bar chart
- Per-class probability distribution

---

##  Docker Instructions

```bash
# Build image
docker build -t cancer-detection .

# Run container
docker run -p 8501:8501 cancer-detection

# Run with GPU support
docker run --gpus all -p 8501:8501 cancer-detection

# With mounted model (after training)
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data/processed:/app/data/processed \
  cancer-detection
```

Open in browser: **http://localhost:8501**

---

##  Results

After training, evaluate results are saved in:

| File | Contents |
|------|----------|
| `data/processed/test_metrics.json` | Accuracy, F1 scores |
| `data/processed/confusion_matrix.png` | Confusion matrix heatmap |
| `data/processed/training_curves.png` | Accuracy/Loss plots |
| `data/processed/classification_report.txt` | Per-class metrics |
| `models/best_model.h5` | Best trained model |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.13 / Keras |
| Transfer Learning | VGG16 (ImageNet) |
| Experiment Tracking | MLflow 2.7 |
| Data Versioning | DVC 3.25 |
| Web App | Streamlit 1.27 |
| Containerization | Docker |
| Data Processing | NumPy, Pandas, OpenCV |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Utilities | Scikit-learn |
| Language | Python 3.9 |

---

##  Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## ⚠️ Medical Disclaimer

This project is for **educational and research purposes only**. It is NOT a medical device and should NOT be used for clinical diagnosis. Always consult a qualified physician.

