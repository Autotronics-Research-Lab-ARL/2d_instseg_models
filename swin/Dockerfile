ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

#install cuda, pytoch, cudnn
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

SHELL ["/bin/bash", "-c"]
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME="/usr/local/cuda-${CUDA}"
ENV FORCE_CUDA="1"

# install os-pkg-deps
RUN apt-get update && apt-get install -y wget nano \
    ffmpeg libsm6 libxext6 git ninja-build \
    libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 \
    -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install Swin-Transformer
RUN conda clean --all
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git /workspace/SWIN/
RUN cd SWIN && pip install -r requirements/build.txt && pip install --no-cache-dir -e .
RUN git clone https://github.com/NVIDIA/apex.git apex/
RUN cd apex && pip install -v --disable-pip-version-check \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install opencv-python==4.4.0.46 timm==0.3.2 termcolor==1.1.0 yacs==0.1.8

WORKDIR SWIN/


