#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


train_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 850 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize
            # 85 normal epoch but 850 save disk size
    fi
}

train_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 850 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
            # 85 normal epoch but 850 save disk size
    fi
}

continue_train(){
    if [ -f $2/model_$5.t7 ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}   \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 850 -nEpochs 41 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber 60 -embSize $embSize
    fi
}

cd ../training


for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR $ALIGNED_DIR
do
    for embSize in 128
    do
        for MODEL_NAME in  "alexnet" "nn4" "vgg-face"  #"alexnet.v2" "nn4-dropout" "vgg-dropout" "nn2"  #"nn4.small1" "nn4.small2"
        do
            for i in crossentropy s_cosine s_hinge t_orj dist_ratio kldiv lmnn s_double_margin t_improved
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/cife/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train_gpu $MODEL $RESULT_DIR $i 30
            done
        done
    done
done
