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

exit -1 # TODO - Add OpenFace nn4.v1 URL below

mkdir -p openface
if [ ! -f openface/nn4.v1.t7 ]; then
  wget TODO \
    -O openface/nn4.v1.t7
fi
