#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"
RESULT_DIR="$PWD/data/results"

if [ ! -d $RAW_DIR ]; then
    python ../util/csv_to_image.py --inputFile fer2013.csv --outputDir $RAW_DIR
fi
if [ ! -d $ALIGNED_DIR/training ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/training --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/publictest ]; then
    python ../util/align-dlib.py $RAW_DIR/publictest align  outerEyesAndNose $ALIGNED_DIR/publictest --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/publictest
fi

if [ ! -d $ALIGNED_DIR/privatetest ]; then
    python ../util/align-dlib.py $RAW_DIR/privatetest align  outerEyesAndNose $ALIGNED_DIR/privatetest --size 48 --rgb 0  \
        --fallbackLfw $RAW_DIR/privatetest
fi


cd ../training

th main.lua -data $ALIGNED_DIR/training -modelDef $WORK_DIR/model/nn4.small2.def.lua -cache $WORK_DIR/data/cache  \
 -save $RESULT_DIR  -nDonkeys 2  -cuda -cudnn -peoplePerBatch 7 -imagesPerPerson 25 -testPy ../evaluation/classify.py \
 -testDir $ALIGNED_DIR/publictest -testBatchSize 100 -epochSize 100 -nEpochs 50 -imgDim 48 -channelSize 1


