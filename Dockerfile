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
    libgl1-mesa-glx \
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
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────
COPY app.py .
COPY src/ ./src/
COPY params.yaml .

# ── Create necessary directories ──────────────────────────
RUN mkdir -p models data/processed data/raw mlruns

# ── Copy pre-trained model if available ───────────────────
# (uncomment after training)
# COPY models/best_model.h5 ./models/
# COPY data/processed/class_mapping.json ./data/processed/

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
