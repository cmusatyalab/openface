FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    python3-opencv \
    wget \
    zip \
    libatlas-base-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libswscale-dev \
    libssl-dev \
    libffi-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD . /root/openface
RUN python3 -m pip install --upgrade pip

WORKDIR /root/openface

RUN ./models/get-models.sh && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install .
#    python3 -m pip install --user --ignore-installed -r demos/web/requirements.txt && \
#    python3 -m pip install -r training/requirements.txt

EXPOSE 8000 9000
