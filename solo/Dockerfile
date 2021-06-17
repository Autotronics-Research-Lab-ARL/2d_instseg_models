ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

#install cuda, pytoch, cudnn
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

SHELL ["/bin/bash", "-c"]
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME="/usr/local/cuda-10.1"
ENV FORCE_CUDA='1'

# install os-pkg-deps
RUN apt-get update && apt-get install -y wget nano \
    ffmpeg libsm6 libxext6 git ninja-build \
    libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install cython pycocotools \
    mmcv==0.2.16 timm==0.3.2 termcolor==1.1.0 yacs==0.1.8

RUN conda install -y -c menpo opencv && conda clean -y --all

# Install SOLO
RUN git clone https://github.com/WXinlong/SOLO.git /workspace/SOLO
RUN cd SOLO && mkdir -p data checkpoints && pip install --no-cache-dir -e .

WORKDIR /workspace/SOLO/
