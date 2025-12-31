FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# -------------------------
# System deps
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# NVIDIA cuDNN
# -------------------------
RUN apt-get update && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    ldconfig

# -------------------------
# Python deps
# -------------------------
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    faster-whisper \
    requests

# -------------------------
# App
# -------------------------
WORKDIR /app
COPY server.py /app/server.py

# -------------------------
# Env
# -------------------------
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV WHISPER_MODEL=GoranS/whisper-base-1m.hr-ctranslate2

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
