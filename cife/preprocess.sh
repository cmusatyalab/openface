#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/train --size 64 --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/test ]; then
    python ../util/align-dlib.py $RAW_DIR/test align  outerEyesAndNose $ALIGNED_DIR/test --size 64 --fallbackLfw $RAW_DIR/test
fi