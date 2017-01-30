# Note from Brandon on 2015-01-13:
#
#   Always push this from an OSX Docker machine.
#
#   If I build this on my Arch Linux desktop it works fine locally,
#   but dlib gives an `Illegal Instruction (core dumped)` error in
#   dlib.get_frontal_face_detector() when running on OSX in a Docker machine.
#   Building in a Docker machine on OSX fixes this issue and the built
#   container successfully deploys on my Arch Linux desktop.
#
# Building this Dockerfile results in the download of approximately 450 MB of data.
#
# Building and pushing:
#   docker build -f opencv-dlib-torch.Dockerfile -t opencv-dlib-torch .
#   docker tag -f <tag of last container> bamos/ubuntu-opencv-dlib-torch:ubuntu_latest-opencv_3.2.0-dlib_19.2-torch_2017.01.29
#   docker push bamos/ubuntu-opencv-dlib-torch:ubuntu_latest-opencv_3.2.0-dlib_19.2-torch_2017.01.29

FROM ubuntu
MAINTAINER Brandon Amos <brandon.amos.cs@gmail.com>

RUN apt-get update && apt-get install -y apt-utils

# Python 3.5
###############################################################################
RUN apt-get install -y python3.5 python3-pip python3.5-dev python3.5-numpy
RUN pip3 install --upgrade pip


# Torch
###############################################################################
RUN apt-get install -y git sudo curl luarocks libqtgui4 libqtcore4 libreadline-dev graphicsmagick libgraphicsmagick1-dev
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch && \
    bash install-deps && \
    ./install.sh && \
    cd install/bin && \
    ./luarocks install nn && \
    ./luarocks install dpnn && \
    ./luarocks install image && \
    ./luarocks install optim && \
    ./luarocks install csvigo && \
    ./luarocks install torchx && \
    ./luarocks install tds && \
    ./luarocks install graphicsmagick && \
    ln -s /root/torch/install/bin/* /usr/local/bin


# OpenCV 3.2
###############################################################################
RUN apt-get install -y zip cmake pkg-config
RUN cd ~ && \
    mkdir -p ocv-tmp && \
    cd ocv-tmp && \
    curl -L https://github.com/Itseez/opencv/archive/3.2.0.zip -o ocv.zip && \
    unzip ocv.zip && \
    cd opencv-3.2.0 && \
    mkdir release && \
    cd release && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_PYTHON_SUPPORT=ON \
          .. && \
    make -j8 && \
    make install && \
    rm -rf ~/ocv-tmp


# Dlib 19.2
###############################################################################
# Note: Python 2.7 is still installed due to libboost-all-dev
RUN apt-get install -y libx11-dev libboost-all-dev
RUN cd ~ && \
    mkdir -p dlib-tmp && \
    cd dlib-tmp && \
    curl -L \
         http://dlib.net/files/dlib-19.2.zip \
         -o dlib.zip && \
    unzip dlib.zip && \
    cd dlib-19.2/examples && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_SSE4_INSTRUCTIONS=ON && \
    cmake --build . --config Release && \
    python3.5 ../../setup.py install --yes USE_AVX_INSTRUCTIONS && \
    rm -rf ~/dlib-tmp
