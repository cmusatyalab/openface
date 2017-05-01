#!/bin/bash

set -x -e

sudo apt-get update
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev \
  libopencv-dev libhdf5-serial-dev libboost-all-dev libgflags-dev \
  libgoogle-glog-dev liblmdb-dev protobuf-compiler libboost-all-dev \
  libatlas-dev libatlas-base-dev liblapack-dev libblas-dev \
  libssl-dev libffi-dev python-pip python-numpy python-imaging \
  python-openssl python-opencv \
  git wget cmake gfortran

mkdir -p ~/src
cd ~/src
git clone https://github.com/bvlc/caffe.git
wget https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2
