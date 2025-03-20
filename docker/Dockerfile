# TensorRT container compatible with CUDA 11.x
FROM nvcr.io/nvidia/tensorrt:21.06-py3

# Set environment variable to disable timezone selection prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10.10
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y tzdata python3.10 python3.10-distutils python3.10-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --set python /usr/bin/python3.10 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Set default timezone
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Update pip and setuptools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy required files
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch (CUDA 11.3 compatible version)
RUN python3.10 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install all dependencies at once
RUN python3.10 -m pip install -r /tmp/requirements.txt

# Set PATH (prioritizing Python 3.10)
ENV PATH="/usr/bin:$PATH"

# Debugging: Check Python version
RUN python --version
RUN python3 --version
RUN which python
RUN which python3

# Configure JupyterLab
RUN python3.10 -m pip install jupyterlab
RUN jupyter lab --generate-config

# Start JupyterLab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
