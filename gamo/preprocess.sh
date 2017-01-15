#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"



ALIGNED_DIR="$PWD/data/aligned64"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi


ALIGNED_DIR="$PWD/data/aligned48"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 48 --rgb 0 --fallbackLfw $RAW_DIR/
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi

ALIGNED_DIR="$PWD/data/notaligned64"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi

ALIGNED_DIR="$PWD/data/notaligned48"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 48 --rgb 0 --fallbackLfw $RAW_DIR/ --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi
