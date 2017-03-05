#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
AUGMENTED_DIR="augmented${imgDim}"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


train ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 750 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize
            # 75 normal epoch but 750 save disk size
    fi
}


cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR $ALIGNED_DIR
do
    for embSize in 128
    do
        for MODEL_NAME in  "alexnet" "nn4" "vgg-face"  #"alexnet.v2" "nn4-dropout" "vgg-dropout" "nn2"  #"nn4.small1" "nn4.small2"
        do
            for i in crossentropy s_cosine s_hinge t_orj dist_ratio kldiv
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train $MODEL $RESULT_DIR $i 30
            done

        done
    done
done

