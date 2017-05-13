#!/usr/bin/env bash

imgDim=224
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
DATA_DIR="raw"
embSize=128

train ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 16 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 500 -nEpochs 50 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

cd ../training


for MODEL_NAME in  nn2
do
    for i in  crossentropy margin s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global
    do
        MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
        RESULT_DIR="$EXTERNAL_DIR/results/imagenet/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
        train $MODEL $RESULT_DIR $i 16
    done
done

