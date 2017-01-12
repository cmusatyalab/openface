#!/usr/bin/env bash

WORK_DIR=$PWD
ALIGNED_DIR="$PWD/data/raw"

train ()
{   if [ ! -d $RESULT_DIR ]; then

        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 10  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 135 -nEpochs 250 -criterion $3 -imgDim 32 -cuda
     fi
}

cd ../training

for MODEL_NAME in "nn4.small1" "nn4.small2" "nn4" "nn2" "vgg-face" "vgg-face.small1"
do
    for i in triplet siamese contrastive
    do
        MODEL=$WORK_DIR/../models/mine/$MODEL_NAME.def.32_1.lua
        RESULT_DIR="$WORK_DIR/data/results_$i/$MODEL_NAME/"

        train $MODEL $RESULT_DIR $i 30

    done
done