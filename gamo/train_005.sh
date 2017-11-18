#!/usr/bin/env bash
export CUDNN_PATH=/home/cenk/cuda5/lib64/libcudnn.so.5

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB1/losses"

train_gpu_005 ()
{
    if [ ! -d $RESULT_DIR ]; then
    echo cache_${imgDim}_$5
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 55 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_005(){
       th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 55 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
    
}

train_gpu_010 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 110 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_010(){
       th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 110 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
    
}


train_gpu_020 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 220 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_020(){
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 220 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
    
}


train_gpu_030 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 330 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_030(){
       th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 330 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
    
}

train_gpu_040 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 440 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_040(){
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 440 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
    
}
train_gpu_050 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 550 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_050(){
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 550 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
 
}
train_gpu_060 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 660 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}

continue_train_060(){
    
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 660 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
 
}
train_gpu_070 ()
{
    if [ ! -d $RESULT_DIR ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5  \
            -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
            -epochSize 770 -nEpochs 500 -imgDim $imgDim -criterion $3 -embSize $embSize
    fi
}
continue_train_070(){
    
    th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache_${imgDim}_$5   \
        -save $2  -nDonkeys 4  -peoplePerBatch 7 -imagesPerPerson $4 -testing  \
        -epochSize 770 -nEpochs 1500 -imgDim $imgDim -criterion $3  \
        -retrain $2/model_500.t7 -epochNumber 501 -embSize $embSize
  
}
cd ../training





continue_train(){

       if [ -f $2/model_$5.t7 ] && [ ! -f $2/model_$6.t7 ]; then
        th main.lua -data $WORK_DIR/data/${INPUT_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}_$9   \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize $8 -nEpochs $7 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber $6 -embSize $embSize
        fi
}

echo data_040_64_128, s_cosine, 151
num=040
i=s_cosine
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 150 151 350 440 $num

echo data_050_64_128, s_cosine, 126
num=050
i=s_cosine
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 125 126 375 550 $num

echo data_050_64_128, s_hadsell,176
num=050
i=s_hadsell
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 175 176 325 550 $num

echo data_050_64_128, t_orj, 175
num=050
i=t_orj
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 175 176 325 550 $num

echo data_060_64_128, s_cosine,101
num=060
i=s_cosine
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 100 101 400 660 $num

echo data_060_64_128, t_orj,475
num=060
i=t_orj
embSize=128
INPUT_DIR="data_${num}_${imgDim}"
MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
continue_train $MODEL $RESULT_DIR $i 30 475 476 25 660 $num

for i in crossentropy t_orj s_cosine s_hadsell histogram
do
    for num in 005 010 020 030 040 050 060 070 080
    do
        INPUT_DIR="data_${num}_${imgDim}"

        for embSize in  128 2 3 4 7
        do
            for MODEL_NAME in nn4
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/gamo_${imgDim}/${INPUT_DIR}_${embSize}/$i/$MODEL_NAME"
                train_gpu_${num} $MODEL $RESULT_DIR $i 30 $num
                #continue_train_${num} $MODEL $RESULT_DIR $i 30
            done

        done
    done
done
