#!/usr/bin/env bash

imgDim=32
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
NOT_ALIGNED_DIR="raw"
ALIGNED_DIR="raw"

train ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/raw/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 100 -imagesPerPerson $4 -testing \
            -epochSize 400 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize

    fi
}


cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR
do
    for MODEL_NAME in nn4
    do
        for i in t_orj
        do
            for embSize in  128 512
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/cifar100/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train $MODEL $RESULT_DIR $i 10
            done
        done
    done
done
