#!/usr/bin/env bash
export CUDNN_PATH=/home/cenk/cuda5/lib64/libcudnn.so.5
imgDim=28
WORK_DIR=$PWD
DATA_DIR=""
EXTERNAL_DIR="/media/cenk/DISK_5TB1/losses"

train ()
{
    if [ ! -d $RESULT_DIR ]; then

        th main.lua -data $WORK_DIR/data/raw/augment -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 600 -nEpochs 250 -imgDim $imgDim -criterion $3 -embSize $embSize

    fi
}


continue_train(){
        if [ -f $2/model_$5.t7 ] && [ ! -f $2/model_$6.t7 ]; then
            th main.lua -data $WORK_DIR/data/raw/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
                -save $2  -nDonkeys 8  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
                -epochSize 600 -nEpochs $8 -imgDim $imgDim -criterion $3 -embSize $embSize \
                -retrain $2/model_$5.t7 -epochNumber $7
        fi
}


cd ../training

for MODEL_NAME in nn4
do
    for i in crossentropy #histogram
    #for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global #lsss histogram
    do
        for embSize in 128
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/fashion_mnist/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
            train $MODEL $RESULT_DIR $i 10
            continue_train $MODEL $RESULT_DIR $i 10 250 275 251 500
        done

    done
done

