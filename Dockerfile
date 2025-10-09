FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy install packages
RUN pip3 install numpy pandas scikit-learn laspy matplotlib requests tqdm pydantic pydantic-settings
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install lazrs[all]


# Copy your code
COPY src/*.py /app/
COPY src/lookup.csv /app/


# Set environment variable for torch cache
ENV TORCH_HOME=/app/torch_cache

# Create cache directory
RUN mkdir -p /app/torch_cache/hub/checkpoints

# Download densenet201 weights
RUN wget -O /app/torch_cache/hub/checkpoints/densenet201-c1103571.pth \
    https://download.pytorch.org/models/densenet201-c1103571.pth

# download the model file
RUN wget -O /app/model_ft_202412171652_3 \
    https://freidata.uni-freiburg.de/records/f850a-bb152/files/model_ft_202412171652_3?download=1

RUN mkdir -p /out && chmod -R 777 /out

RUN python3 -c "import torch; print(torch.cuda.is_available()); import time; time.sleep(2.5)"

# Set entrypoint
CMD ["python3", "run.py"]