#!/usr/bin/env bash

imgDim=48
WORK_DIR=$PWD
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"

train ()
{
    if [ ! -d $RESULT_DIR ]; then
       th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 240 -nEpochs 50 -imgDim $imgDim -criterion $3 -embSize $embSize -channelSize 1
    fi
}

continue_train(){
    if [ -f $2/model_$5.t7 ]; then

        th main.lua -data $DATA_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 240 -nEpochs 50 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber 51 -embSize $embSize -channelSize 1
    fi
}

cd ../training
for DATA_DIR in $NOT_ALIGNED_DIR #$ALIGNED_DIR
do
    for embSize in 128
    do
        for i in triplet contrastive siamese
        do
            for MODEL_NAME in  "alexnet"  "vgg-face" #"alexnet.v2" "nn4" "nn2"  #"nn4.small1" "nn4.small2"
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$WORK_DIR/results/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train $MODEL $RESULT_DIR $i 30
            done
        done
        for i in contrastive triplet siamese
        do
            for MODEL_NAME in  "alexnet" "vgg-face" #"alexnet.v2"  "nn4" "nn2"  #"nn4.small1" "nn4.small2"
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$WORK_DIR/results/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

                continue_train $MODEL $RESULT_DIR $i 30 50
                # model_path, result_path, cost_function, imagePerPerson, retrain_from


            done

        done
    done
done