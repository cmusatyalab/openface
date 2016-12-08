#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"
RESULT_DIR="$PWD/data/results"


if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/train --size 64 --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/test ]; then
    python ../util/align-dlib.py $RAW_DIR/test align  outerEyesAndNose $ALIGNED_DIR/test --size 64 --fallbackLfw $RAW_DIR/test
fi


if [ ! -d $RESULT_DIR ]; then
cd ../training

th main.lua -data $ALIGNED_DIR/train -modelDef $WORK_DIR/model/nn4.small2.def.lua -cache $WORK_DIR/data/cache  \
 -save $RESULT_DIR  -nDonkeys 2 -peoplePerBatch 7 -imagesPerPerson 25 -testPy ../evaluation/classify.py \
 -testDir $ALIGNED_DIR/test -testBatchSize 100 -epochSize 1000 -nEpochs 100 -imgDim 64
fi



WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"
RESULT_DIR="$PWD/data/results_vgg"

if [ ! -d $RESULT_DIR ]; then
cd ../training
th main.lua -data $ALIGNED_DIR/train -modelDef $WORK_DIR/model/vgg-face.small1.def.lua -cache $WORK_DIR/data/cache  \
 -save $RESULT_DIR  -nDonkeys 2  -peoplePerBatch 7 -imagesPerPerson 25 -testPy ../evaluation/classify.py \
 -testDir $ALIGNED_DIR/test -testBatchSize 100 -epochSize 500 -nEpochs 100 -imgDim 64
fi