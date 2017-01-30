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
# Building and pushing:
#   docker build -f opencv-dlib-torch.Dockerfile -t opencv-dlib-torch .
#   docker tag -f <tag of last container> bamos/ubuntu-opencv-dlib-torch:ubuntu_latest-opencv_3.2.0-dlib_19.2-torch_2017.01.29
#   docker push bamos/ubuntu-opencv-dlib-torch:ubuntu_latest-opencv_3.2.0-dlib_19.2-torch_2017.01.29

FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_latest-opencv_3.2.0-dlib_19.2-torch_2017.01.29
MAINTAINER Brandon Amos <brandon.amos.cs@gmail.com>

# OpenFace
###############################################################################
RUN cd ~ && \
    git clone https://github.com/qacollective/openface.git ~/openface --recursive && \
    cd ~/openface && \
    ./models/get-models.sh && \
    pip3 install -r requirements.txt && \
    python3.5 setup.py install && \
    pip3 install -r demos/web/requirements.txt && \
    pip3 install -r training/requirements.txt

EXPOSE 8000 9000
CMD /bin/bash -l -c '/root/openface/demos/web/start-servers.sh'
