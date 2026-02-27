# ============================================================
# TT - Financial Analysis & ML Pipeline
# Docker Image with CUDA 12.8 support
# ============================================================

# --- Stage 1: Build TA-Lib from source ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Compile TA-Lib C library
WORKDIR /tmp
RUN wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install

# --- Stage 2: Final image ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy TA-Lib from builder
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib /usr/local/include/ta-lib
RUN ldconfig

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Install PyTorch with CUDA first (from PyTorch index)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
RUN pip install --no-cache-dir --break-system-packages \
    -r requirements.txt \
    --ignore-installed torch torchvision torchaudio

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
