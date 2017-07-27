#!/usr/bin/env bash

train_gamo_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_gamo_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
    fi
}


train_cife_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_cife_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
    fi
}


train_mnist_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}


train_mnist_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 1 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
    fi
}


train_cifar10_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 2 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

train_cifar10_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 10 -imagesPerPerson $4 -testing \
            -epochSize 1 -nEpochs 1 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
    fi
}

cd training

imgDim=64
WORK_DIR=$PWD/../gamo
EXTERNAL_DIR="/media/cenk/DISK_5TB/epoch_times_gpu"
DATA_DIR="notaligned${imgDim}"

for embSize in 128
do
    for MODEL_NAME in  alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

            train_gamo_gpu $MODEL $RESULT_DIR $i 30
        done
    done
done



imgDim=64
WORK_DIR=$PWD/../cife
DATA_DIR="notaligned${imgDim}"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/cife/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_cife_gpu $MODEL $RESULT_DIR $i 30
        done
    done
done



imgDim=28
WORK_DIR=$PWD/../mnist
DATA_DIR="raw"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/mnist/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_mnist_gpu $MODEL $RESULT_DIR $i 10
        done
    done
done


imgDim=32
WORK_DIR=$PWD/../cifar10
DATA_DIR="raw"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/cifar10/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_cifar10_gpu $MODEL $RESULT_DIR $i 10

        done
    done
done



imgDim=64
WORK_DIR=$PWD/../gamo
EXTERNAL_DIR="/media/cenk/DISK_5TB/epoch_times_cpu"
DATA_DIR="notaligned${imgDim}"

for embSize in 128
do
    for MODEL_NAME in  alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

            train_gamo_cpu $MODEL $RESULT_DIR $i 30
        done
    done
done



imgDim=64
WORK_DIR=$PWD/../cife
DATA_DIR="notaligned${imgDim}"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/cife/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_cife_cpu $MODEL $RESULT_DIR $i 30
        done
    done
done



imgDim=28
WORK_DIR=$PWD/../mnist
DATA_DIR="raw"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/mnist/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_mnist_cpu $MODEL $RESULT_DIR $i 10
        done
    done
done


imgDim=32
WORK_DIR=$PWD/../cifar10
DATA_DIR="raw"


for embSize in 128
do
    for MODEL_NAME in alexnet nn4 vgg-face
    do
        for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global  lsss
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/cifar10/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
            train_cifar10_cpu $MODEL $RESULT_DIR $i 10

        done
    done
done