#!/usr/bin/env bash

imgDim=112
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB"
RAW_DIR="raw"

train ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 16  -peoplePerBatch 2 -imagesPerPerson $4 -testing \
            -epochSize 540 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize 128
    fi
}


cd ../training

for DATA_DIR in $RAW_DIR
do
    for i in contrastive
    do
        for MODEL_NAME in  "vgg-face"
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/disfa/${DATA_DIR}/$i/$MODEL_NAME"
            # model_path, result_path, cost_function, imagePerPerson
            train $MODEL $RESULT_DIR $i 50
        done
    done

done