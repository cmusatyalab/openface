#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 80  -peoplePerBatch 7 -imagesPerPerson 100 -testPy ../evaluation/classify.py -testing \
            -testDir $ALIGNED_DIR/test -testBatchSize 100 -epochSize 100 -nEpochs 50 -imgDim 64 -criterion cosine -alpha $3

    fi
}

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi

cd ../training


for i in 0.25 0.5
do

    MODEL=$WORK_DIR/../models/mine/nn4.small3.def.64_1.lua
    RESULT_DIR="$WORK_DIR/data/results_cosine/nn4.small3/alpha$i"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i
    fi
#
#    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/nn4.small2/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/nn4.small1.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/nn4.small1/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
#
#
#    MODEL=$WORK_DIR/../models/mine/nn2.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/nn2/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/nn4.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/nn4/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/vgg-face.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/vgg/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/vgg-face.small1.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_cosine/vgg.small1/alpha$i"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR $i
#    fi
done