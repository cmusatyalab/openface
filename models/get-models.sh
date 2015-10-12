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

set +x
echo ==========
echo The nn4.v1.t7 and celeb-classifier.nn4.v1.pkl models are
echo copyright Carnegie Mellon University and are licensed under
echo the Apache 2.0 License.
echo ==========
set -x
mkdir -p openface
if [ ! -f openface/nn4.v1.t7 ]; then
  wget http://openface-models.storage.cmusatyalab.org/nn4.v1.t7 \
    -O openface/nn4.v1.t7
  wget http://openface-models.storage.cmusatyalab.org/celeb-classifier.nn4.v1.pkl \
    -O openface/celeb-classifier.nn4.v1.pkl
fi
