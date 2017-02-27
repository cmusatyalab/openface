#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/200"
AUGMENTED_DIR="augmented${imgDim}"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


train ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 750 -nEpochs 59 -imgDim $imgDim -criterion $3 -embSize $embSize
            # 75 normal epoch but 750 save disk size
    fi
}

continue_train(){
    if [ -f $2/model_$5.t7 ]; then

        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}   \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 750 -nEpochs 16 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber 185 -embSize $embSize
    fi
}

cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR # #$ALIGNED_DIR
do
    for embSize in 32
    do
        for i in  siamese
        do
            for MODEL_NAME in  "nn4" #"vgg-face" "alexnet" #"alexnet.v2" "nn4-dropout" "vgg-dropout" "nn4" "nn2"  #"nn4.small1" "nn4.small2"
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson

                continue_train $MODEL $RESULT_DIR $i 30 184
            done

        done
    done
done

