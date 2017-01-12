#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache48  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 2400 -nEpochs 100 -criterion $3 -imgDim 48 -channelSize 1
    fi
}

cd ../training

for i in triplet siamese contrastive
do
    for MODEL_NAME in "nn4.small1" "nn4.small2" "nn4" "nn2" "vgg-face" "vgg-face.small1" "alexnet"
    do
        MODEL=$WORK_DIR/../models/mine/$MODEL_NAME.def.48_1.lua
        RESULT_DIR="$WORK_DIR/results/$i/$MODEL_NAME"
        # model_path, result_path, cost_function, imagePerPerson
        train $MODEL $RESULT_DIR $i 30
    done

done