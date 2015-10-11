#!/bin/bash
#
# Download OpenFace models.

cd "$(dirname "$0")"

set -x -e

mkdir -p dlib
if [ ! -f dlib/shape_predictor_68_face_landmarks.dat ]; then
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    -O dlib/shape_predictor_68_face_landmarks.dat.bz2
  bunzip2 dlib/shape_predictor_68_face_landmarks.dat.bz2
fi

mkdir -p openface
if [ ! -f openface/nn4.v1.t7 ]; then
  wget http://openface-models.storage.cmusatyalab.org/nn4.v1.t7 \
    -O openface/nn4.v1.t7
  wget http://openface-models.storage.cmusatyalab.org/celeb-classifier.nn4.v1.pkl \
    -O openface/celeb-classifier.nn4.v1.pkl
fi
