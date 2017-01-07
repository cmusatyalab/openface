#!/usr/bin/env bash

WORK_DIR=$PWD
ALIGNED_DIR="$PWD/data/raw"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 10  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 500 -nEpochs 250 -criterion $3 -imgDim 32 -cuda
    fi
}

cd ../training

for i in triplet kl
do
    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.32_1.lua
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i 25
    fi
done

for i in loglikelihood cosine l2loss
do
    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.32_1.lua
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i 50
    fi
done