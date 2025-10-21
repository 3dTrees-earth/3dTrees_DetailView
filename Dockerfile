# Build stage (optional, if you need to compile wheels)
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04 AS builder

WORKDIR /app

# Install python and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && pip3 install --no-cache-dir -r requirements.txt

# ---

# Final stage: minimal runtime
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install python and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev libgl1 libglx-mesa0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code and required assets only
COPY src/*.py /app/
COPY src/lookup.csv /app/

# ENV and cache setup
ENV TORCH_HOME=/app/torch_cache
RUN mkdir -p /app/torch_cache/hub/checkpoints

# Download model weights (delete wget after)
RUN apt-get update && apt-get install -y wget && \
    wget -O /app/torch_cache/hub/checkpoints/densenet201-c1103571.pth \
    https://download.pytorch.org/models/densenet201-c1103571.pth && \
    wget -O /app/model_ft_202412171652_3 \
    https://freidata.uni-freiburg.de/records/f850a-bb152/files/model_ft_202412171652_3?download=1 && \
    apt-get purge -y wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create input/output directories
RUN mkdir -p /out && chmod -R 777 /out && \
    mkdir -p /in && chmod -R 777 /in

# Test torch
RUN python3 -c "import torch; print(torch.cuda.is_available()); import time; time.sleep(2.5)"

ENTRYPOINT ["python3", "run.py"]