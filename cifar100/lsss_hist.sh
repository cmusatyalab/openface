#!/usr/bin/env bash

imgDim=32
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
NOT_ALIGNED_DIR="raw"
ALIGNED_DIR="raw"
embSize=128
train ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/raw/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 10 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/raw/test \
            -epochSize 4000 -nEpochs 50 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda

    fi
}



cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in lsss histogram
        do
            for embSize in 128
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/cifar10/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
                train $MODEL $RESULT_DIR $i 10
            done
        done
    done
done
