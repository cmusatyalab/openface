#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 80  -peoplePerBatch 7 -imagesPerPerson 10 -testing \
            -epochSize 500 -nEpochs 50 -criterion $3 -imgDim 48 -channelSize 1 -cuda
    fi
}

cd ../training

for i in loglikelihood triplet  #l1hinge #cosine
do
    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.48_1.lua
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i
    fi
done

#    MODEL=$WORK_DIR/../models/mine/nn4.small3.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small3/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi
#
#
#    MODEL=$WORK_DIR/../models/mine/nn4.small1.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small1/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi
#
#
#    MODEL=$WORK_DIR/../models/mine/nn2.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn2/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/nn4.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/vgg-face.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/vgg/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi
#
#    MODEL=$WORK_DIR/../models/mine/vgg-face.small1.def.64_1.lua
#    RESULT_DIR="$WORK_DIR/data/results_triplet/vgg.small1/"
#    if [ ! -d $RESULT_DIR ]; then
#
#        train $MODEL $RESULT_DIR
#    fi