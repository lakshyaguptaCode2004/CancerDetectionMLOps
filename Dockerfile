# =============================================================
# Dockerfile
# =============================================================
# Chest Cancer Detection - Streamlit App
#
# Build:
#   docker build -t cancer-detection .
#
# Run:
#   docker run -p 8501:8501 cancer-detection
#
# With GPU:
#   docker run --gpus all -p 8501:8501 cancer-detection
# =============================================================

# ── Base image: Python 3.9 slim ───────────────────────────
FROM python:3.9-slim

# ── Metadata ──────────────────────────────────────────────
LABEL maintainer="CancerDetectionMLOps"
LABEL description="Chest Cancer Classification using CNN"
LABEL version="1.0.0"

# ── Environment variables ─────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── System dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow==2.13.0 \
        numpy==1.24.3 \
        pillow==9.5.0 && \
    pip install --no-cache-dir \
        pandas==2.0.3 \
        matplotlib==3.7.2 \
        seaborn==0.12.2 \
        scikit-learn==1.3.0 \
        plotly==5.17.0 && \
    pip install --no-cache-dir \
        mlflow==2.7.1 \
        grpcio==1.57.0 \
        protobuf==4.23.4 && \
    pip install --no-cache-dir \
        streamlit==1.27.0 && \
    pip install --no-cache-dir \
        opencv-python-headless==4.8.1.78 \
        pyyaml==6.0.1
```

Also update `requirements.txt` — replace entire contents with:
```
tensorflow==2.13.0
numpy==1.24.3
pillow==9.5.0
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
mlflow==2.7.1
grpcio==1.57.0
protobuf==4.23.4
streamlit==1.27.0
plotly==5.17.0
pyyaml==6.0.1
opencv-python-headless==4.8.1.78
Notice dvc and ipykernel are removed — they're development tools not needed inside the Docker container at all. The app only needs tensorflow, streamlit, mlflow, and supporting libs.

Then rebuild:

cmd
docker build --no-cache -t cancer-detection .
If it still takes too long or fails, use this even simpler minimal Dockerfile that bypasses all the conflict issues entirely:

dockerfile
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow==2.13.0 numpy==1.24.3 pillow==9.5.0

RUN pip install --no-cache-dir \
    pandas==2.0.3 matplotlib==3.7.2 scikit-learn==1.3.0 \
    plotly==5.17.0 pyyaml==6.0.1 opencv-python-headless==4.8.1.78

RUN pip install --no-cache-dir \
    grpcio==1.57.0 protobuf==4.23.4

RUN pip install --no-cache-dir streamlit==1.27.0

RUN pip install --no-cache-dir mlflow==2.7.1

COPY app.py .
COPY src/ ./src/
COPY params.yaml .
RUN mkdir -p models data/processed data/raw mlruns
COPY models/best_model.h5 ./models/
COPY data/processed/class_mapping.json ./data/processed/

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
The key insight here is splitting every pip install into separate RUN layers — each layer caches independently so if one fails you don't restart from zero, and pip's resolver works on a smaller graph each time.





# ── Copy application code ─────────────────────────────────
COPY app.py .
COPY src/ ./src/
COPY params.yaml .

# ── Create necessary directories ──────────────────────────
RUN mkdir -p models data/processed data/raw mlruns

# ── Copy pre-trained model if available ───────────────────
# (uncomment after training)
COPY models/best_model.h5 ./models/
COPY data/processed/class_mapping.json ./data/processed/

# ── Expose Streamlit port ─────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Launch Streamlit app ──────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
