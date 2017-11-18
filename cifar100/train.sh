#!/usr/bin/env bash
export CUDNN_PATH=/home/cenk/cuda5/lib64/libcudnn.so.5

imgDim=32
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB1/losses"
NOT_ALIGNED_DIR="raw"
ALIGNED_DIR="raw"

train ()
{
    if [ ! -d $RESULT_DIR ]; then

        th main.lua -data $WORK_DIR/data/raw/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 50 -imagesPerPerson $4 -testing \
            -epochSize 100 -nEpochs 1000 -imgDim $imgDim -criterion $3 -embSize $embSize

    fi
}


cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR
do
    for MODEL_NAME in nn4
    do
        for i in crossentropy t_orj s_cosine s_hadsell histogram
        do
            for embSize in 256
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/cifar100/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train $MODEL $RESULT_DIR $i 10
            done
        done
    done
done
