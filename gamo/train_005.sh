#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB1/losses"





train_gpu_005 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 55 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_010 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 110 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_015 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 165 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_020 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 220 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}
train_gpu_025 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 275 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_030 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 330 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_040 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 440 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_050 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 550 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_060 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 660 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gpu_070 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 770 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}
cd ../training

for i in crossentropy t_orj s_cosine s_hadsell histogram
do
    for num in 005 010 020 030 040 050 060 070 080
    do
        INPUT_DIR="data_${num}_${imgDim}"


        for DATA_DIR in $INPUT_DIR
        do
            for embSize in 128
            do
                for MODEL_NAME in nn4
                do
                    MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                    RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                    train_gpu_${num} $MODEL $RESULT_DIR $i 30
                done

            done
        done
    done
done
