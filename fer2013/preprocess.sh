#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

#PREPROCESS
if [ ! -d $RAW_DIR ]; then
    python ../util/csv_to_image.py --inputFile fer2013.csv --outputDir $RAW_DIR
fi
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/training --size 64 --rgb 1 \
        --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/test ]; then
    python ../util/align-dlib.py $RAW_DIR/publictest align  outerEyesAndNose $ALIGNED_DIR/publictest --size 64 --rgb 1 \
        --fallbackLfw $RAW_DIR/publictest
fi

if [ ! -d $ALIGNED_DIR/privatetest ]; then
    python ../util/align-dlib.py $RAW_DIR/privatetest align  outerEyesAndNose $ALIGNED_DIR/privatetest --size 64 --rgb 1  \
        --fallbackLfw $RAW_DIR/privatetest
fi

mv $ALIGNED_DIR/training $ALIGNED_DIR/train
mv $ALIGNED_DIR/publictest $ALIGNED_DIR/test



ALIGNED_DIR="$PWD/data/aligned48"

#PREPROCESS
if [ ! -d $RAW_DIR ]; then
    python ../util/csv_to_image.py --inputFile fer2013.csv --outputDir $RAW_DIR
fi
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/training --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/test ]; then
    python ../util/align-dlib.py $RAW_DIR/publictest align  outerEyesAndNose $ALIGNED_DIR/publictest --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/publictest
fi

if [ ! -d $ALIGNED_DIR/privatetest ]; then
    python ../util/align-dlib.py $RAW_DIR/privatetest align  outerEyesAndNose $ALIGNED_DIR/privatetest --size 48 --rgb 0  \
        --fallbackLfw $RAW_DIR/privatetest
fi

mv $ALIGNED_DIR/training $ALIGNED_DIR/train
mv $ALIGNED_DIR/publictest $ALIGNED_DIR/test