#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/train --size 64 --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/test ]; then
    python ../util/align-dlib.py $RAW_DIR/test align  outerEyesAndNose $ALIGNED_DIR/test --size 64 --fallbackLfw $RAW_DIR/test
fi


train ()
{

    th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
        -save $2  -nDonkeys 80  -peoplePerBatch 7 -imagesPerPerson 15 -testPy ../evaluation/classify.py \
        -testDir $ALIGNED_DIR/test -testBatchSize 1 -epochSize 100 -nEpochs 20 -imgDim 64 -cuda -cudnn
}


cd ../training

MODEL=$WORK_DIR/../models/mine/nn2.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/nn2"
echo $RESULT_DIR
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/nn4"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.small1.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/nn4.small1"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.small2.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/nn4.small2"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/vgg-face.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/vgg"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/vgg-face.small1.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results/vgg.small1"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi