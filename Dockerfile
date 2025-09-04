FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-glx \
    libegl1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV TORCH_HOME=/app/torch_cache
ENV EGL_PLATFORM=surfaceless

# Create directories
RUN mkdir -p /app/torch_cache/hub/checkpoints
RUN mkdir -p /src && mkdir -p /in && mkdir -p /out

# Set work directory
WORKDIR /app

# Install Python packages
RUN pip3 install \
    numpy \
    pydantic \
    pydantic-settings \
    tqdm \
    pandas \
    scikit-learn \
    laspy \
    matplotlib \
    requests \
    fastapi \
    uvicorn

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy your code
COPY *.py /app/
COPY /src/lookup.csv /app/

# Copy source code to /src as well
COPY /src/ /src

# Download densenet201 weights
RUN wget -O /app/torch_cache/hub/checkpoints/densenet201-c1103571.pth \
    https://download.pytorch.org/models/densenet201-c1103571.pth

# Install laspy with lazrs backend for LAZ support
RUN pip3 install "laspy[lazrs]"

RUN python3 -c "import torch; print(torch.cuda.is_available()); import time; time.sleep(2.5)"

# Set working directory for execution
WORKDIR /app

# set permissions for all folders
# Set proper permissions for all directories
RUN chmod -R 755 /app && \
    chmod -R 755 /src && \
    chmod -R 755 /in && \
    chmod -R 755 /out && \
    chmod -R 777 /tmp && \
    chmod -R 755 /app/torch_cache


# Set entrypoint
CMD ["python3", "run.py"]
# ENTRYPOINT ["python3", "predict.py"]
# ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t detailview .
# docker run --rm --gpus all -v "C:/TLS/docker/input:/input" -v "C:/TLS/docker//output:/output" detailview --prediction_data /input/circle_3_segmented.las --model_path /app/model_ft_202412171652_3