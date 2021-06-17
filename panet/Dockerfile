ARG PYTORCH="0.4"
ARG CUDA="9"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

SHELL ["/bin/bash", "-c"]
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA='1'

RUN apt-get update && apt-get upgrade -y \ 
    && apt-get update && apt-get install -y wget nano \
    ffmpeg libsm6 libxext6 git ninja-build cython\
    libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install cython matplotlib numpy==1.15 scipy \ 
    gdown pyyaml packaging pycocotools tensorboardX

RUN echo "cudatoolkit =9.0" > /opt/conda/conda-meta/pinned \
    && echo "pytorch =0.4.0" >> /opt/conda/conda-meta/pinned \
    && echo "torchvision =0.2.1" >> /opt/conda/conda-meta/pinned \
    && echo "setuptools =39.2.0" >> /opt/conda/conda-meta/pinned \  
    && echo "pip =10.0.1" >> /opt/conda/conda-meta/pinned \  
    && echo "python =3.6.5" >> /opt/conda/conda-meta/pinned \  
    && echo "conda =4.5.4" >> /opt/conda/conda-meta/pinned \
    && echo "wheel =0.31.1" >> /opt/conda/conda-meta/pinned

RUN conda install -c menpo opencv && conda clean -y --all
RUN git clone https://github.com/ShuLiu1993/PANet.git /workspace/PANet
RUN cd PANet && mkdir -p data/test data/pretrained_model && cd lib && bash make.sh

WORKDIR /workspace/PANet
