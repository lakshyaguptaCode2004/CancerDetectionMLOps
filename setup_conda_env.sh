#!/bin/bash
# =============================================================
# setup_conda_env.sh
# =============================================================
# One-click setup for Chest Cancer Detection MLOps project
# Creates a dedicated conda environment with all dependencies
#
# Usage:
#   chmod +x setup_conda_env.sh
#   ./setup_conda_env.sh
# =============================================================

set -e  # Exit on error

ENV_NAME="cancer_detection"
PYTHON_VERSION="3.9"

echo ""
echo "======================================================"
echo "  Chest Cancer Detection MLOps - Environment Setup"
echo "======================================================"
echo ""

# ── Check conda ────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda or Anaconda:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ conda found: $(conda --version)"

# ── Remove existing env if present ────────────────────────
if conda env list | grep -q "^${ENV_NAME}"; then
    echo ""
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "   Remove and recreate? (y/n): " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        echo "   Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "   Using existing environment."
        conda activate "${ENV_NAME}"
        echo "✅ Activated: ${ENV_NAME}"
        exit 0
    fi
fi

# ── Create conda environment ───────────────────────────────
echo ""
echo "Creating conda environment: '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# ── Activate environment ───────────────────────────────────
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Install pip packages ───────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install --upgrade pip

pip install \
    tensorflow==2.13.0 \
    keras==2.13.1 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    mlflow==2.7.1 \
    streamlit==1.27.0 \
    dvc==3.25.0 \
    pillow==10.0.1 \
    opencv-python-headless==4.8.1.78 \
    plotly==5.17.0 \
    pyyaml==6.0.1 \
    nbformat==5.9.2 \
    ipykernel==6.25.2 \
    jupyter

# ── Register Jupyter kernel ────────────────────────────────
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (cancer_detection)"

# ── Initialize DVC ─────────────────────────────────────────
echo ""
echo "Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "✅ DVC initialized"
else
    echo "   DVC already initialized"
fi

# ── Create data directories ────────────────────────────────
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed models mlruns notebooks src

# ── Print DVC track commands ───────────────────────────────
echo ""
echo "To track dataset with DVC:"
echo "  dvc add data/raw"
echo "  git add data/raw.dvc .gitignore"
echo "  git commit -m 'Track dataset with DVC'"

# ── Summary ────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  ✅ Setup Complete!"
echo "======================================================"
echo ""
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Download dataset:"
echo "  kaggle datasets download -d mohamedhanyyy/chest-ctscan-images"
echo "  unzip chest-ctscan-images.zip -d data/raw/"
echo ""
echo "Run pipeline:"
echo "  python src/data_ingestion.py"
echo "  python src/preprocessing.py"
echo "  python src/train_cnn.py"
echo "  python src/evaluate.py"
echo ""
echo "Or run all at once with DVC:"
echo "  dvc repro"
echo ""
echo "Launch Streamlit app:"
echo "  streamlit run app.py"
echo ""
echo "Launch MLflow UI:"
echo "  mlflow ui --backend-store-uri mlruns"
echo "======================================================"
