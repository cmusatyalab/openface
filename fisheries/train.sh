#!/usr/bin/env bash

imgDim=400
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"


train_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 32  -peoplePerBatch 8 -imagesPerPerson $4 -testBatchSize 5  -testDir $WORK_DIR/data/test \
            -epochSize 50 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize -testing
            # 75 normal epoch but 750 save disk size
    fi
}



cd ../training



for embSize in 128
do
    for MODEL_NAME in  "alexnet"  #"alexnet.v2" "nn4-dropout" "vgg-dropout" "nn2"  #"nn4.small1" "nn4.small2"
    do
        for i in crossentropy t_orj
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/fisheries/${embSize}/$i/$MODEL_NAME"
            # model_path, result_path, cost_function, imagePerPerson
            train_cpu $MODEL $RESULT_DIR $i 10
        done

    done
done