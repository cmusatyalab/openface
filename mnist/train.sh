#!/usr/bin/env bash

imgDim=28
WORK_DIR=$PWD
DATA_DIR="raw"
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"

train ()
{
    if [ ! -d $RESULT_DIR ]; then

        th main.lua -data $WORK_DIR/data/raw/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 10 -imagesPerPerson $4 -testBatchSize 10  -testDir $WORK_DIR/data/raw/test \
            -epochSize 600 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize

    fi
}
# 'crossentropy' 'kldiv'
# 's_cosine' 's_hinge' 's_double_margin' 's_global'
# 't_orj' 't_improved' 't_global' 'dist_ratio'
# 'lsss' 'lmnn' 'softPN' 'histogram' 'quadruplet'

cd ../training


for MODEL_NAME in alexnet nn4 vgg-face
do
    for i in crossentropy t_orj dist_ratio s_cosine s_hinge kldiv
    do
        for embSize in 128
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/mnist/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
            # model_path, result_path, cost_function, imagePerPerson
            train $MODEL $RESULT_DIR $i 10
            sh $WORK_DIR/test.sh
        done

    done
done

