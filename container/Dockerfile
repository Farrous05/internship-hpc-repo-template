FROM nvcr.io/nvidia/pytorch:25.03-py3
WORKDIR /workspace

# Install build tools and speed-ups
RUN apt-get update && apt-get install -y \
    python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Fast build tool for FlashAttention
RUN pip install ninja packaging

# install transformers and datasets
RUN pip install transformers datasets accelerate peft

# Install FlashAttention (compatible with PyTorch >= 2.2, CUDA >= 12.0)
RUN pip install flash-attn --no-build-isolation
