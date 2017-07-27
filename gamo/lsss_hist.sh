#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
AUGMENTED_DIR="augmented${imgDim}"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


train_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 750 -nEpochs 100 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}


cd ../training




for DATA_DIR in $NOT_ALIGNED_DIR
do
    for embSize in 128
    do
        for MODEL_NAME in  alexnet nn4 vgg-face
        do
            for i in lsss histogram
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                train_cpu $MODEL $RESULT_DIR $i 30
            done

        done
    done
done

