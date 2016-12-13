#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 80  -peoplePerBatch 7 -imagesPerPerson 50 -testPy ../evaluation/classify.py -testing \
            -testDir $ALIGNED_DIR/test -testBatchSize 100 -epochSize 51 -nEpochs 250 -imgDim 64 -cuda -cudnn -criterion loglikelihood
    fi
}

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi

cd ../training


MODEL=$WORK_DIR/../models/mine/nn4.small3.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small3"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.small2.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small2"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.small1.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small1"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi


MODEL=$WORK_DIR/../models/mine/nn2.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn2"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/nn4.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn4"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/vgg-face.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/vgg"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi

MODEL=$WORK_DIR/../models/mine/vgg-face.small1.def.64_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/vgg.small1"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi