#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 750 -nEpochs 50 -imgDim 64 -criterion $3
            # 75 normal epoch but 750 save disk size
    fi
}

continue_train(){
    if [-f $2/model_50.t7]; then

        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 750 -nEpochs 50 -imgDim 64 -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber 51
    fi
}
cd ../training
for i in triplet siamese contrastive
do
    for MODEL_NAME in "nn4.small1" "nn4.small2" "nn4" "nn2" "vgg-face" "vgg-face.small1" "alexnet"
    do
        MODEL=$WORK_DIR/../models/mine/$MODEL_NAME.def.64_1.lua
        RESULT_DIR="$WORK_DIR/results/$i/$MODEL_NAME"

        # model_path, result_path, cost_function, imagePerPerson
        train $MODEL $RESULT_DIR $i 30

        continue_train $MODEL $RESULT_DIR $i 30 50
        # model_path, result_path, cost_function, imagePerPerson, retrain_from
    done

done