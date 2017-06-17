#!/usr/bin/env bash

imgDim=96
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
embSize=128

train ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 32  -peoplePerBatch 16 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/test \
            -epochSize 7100 -nEpochs 2000 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

contuine ()
{

        th main.lua -data $WORK_DIR/data/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 32  -peoplePerBatch 16 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/test \
            -epochSize 7100 -nEpochs 2000 -imgDim $imgDim -criterion $3 -embSize $embSize -retrain 90 -epochNumber 91

}

cd ../training


for MODEL_NAME in nn4
do
    for i in  crossentropy t_orj s_cosine
    do
        MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
        RESULT_DIR="$EXTERNAL_DIR/results/imagenet/${embSize}/$i/$MODEL_NAME"

        train $MODEL $RESULT_DIR $i 16
    done
done

