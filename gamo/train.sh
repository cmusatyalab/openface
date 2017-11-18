#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
AUGMENTED_DIR="augmented${imgDim}"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


train_gpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 750 -nEpochs 50 -imgDim $imgDim -criterion $3 -embSize $embSize
            # 75 normal epoch but 750 save disk size
    fi
}

train_cpu ()
{
    if [ ! -d $RESULT_DIR ]; then


        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 750 -nEpochs 200 -imgDim $imgDim -criterion $3 -embSize $embSize -cuda
            # 75 normal epoch but 750 save disk size
    fi
}


cd ../training


continue_train(){
echo $2
echo model_$5.t7
    if [ -f $2/model_$5.t7 ] && [ ! -f $2/model_$6.t7 ]; then
        th main.lua -data $WORK_DIR/data/${DATA_DIR}/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}   \
            -save $2  -nDonkeys 16  -peoplePerBatch 7 -imagesPerPerson $4 -testBatchSize 50  -testDir $WORK_DIR/data/${DATA_DIR}/test \
            -epochSize 750 -nEpochs $7 -imgDim $imgDim -criterion $3  \
            -retrain $2/model_$5.t7 -epochNumber $6 -embSize $embSize
    fi
}





example_continue(){

DATA_DIR=$NOT_ALIGNED_DIR

MODEL_NAME=alexnet
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
i=lsss
embSize=128
RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
# model_path, result_path, cost_function, imagePerPerson
continue_train $MODEL $RESULT_DIR $i 30 61 62 39


MODEL_NAME=nn4
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
i=lsss
embSize=128
RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
# model_path, result_path, cost_function, imagePerPerson
continue_train $MODEL $RESULT_DIR $i 30 10 11 90

MODEL_NAME=vgg-face
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
i=s_hadsell
embSize=128
RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
# model_path, result_path, cost_function, imagePerPerson
continue_train $MODEL $RESULT_DIR $i 30 28 29 72

MODEL_NAME=vgg-face
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
i=t_orj
embSize=128
RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
# model_path, result_path, cost_function, imagePerPerson
continue_train $MODEL $RESULT_DIR $i 30 68 69 32

MODEL_NAME=alexnet
MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
i=histogram
embSize=128
RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
# model_path, result_path, cost_function, imagePerPerson
continue_train $MODEL $RESULT_DIR $i 30 16 17 84


}

#example_continue

for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR $ALIGNED_DIR
do
    for embSize in 128
    do
        for MODEL_NAME in  alexnet nn4 vgg-face
        do
            for i in t_entropy multi margin crossentropy s_cosine s_hinge t_orj dist_ratio kldiv t_improved s_hadsell s_double_margin lmnn softPN t_global #lsss histogram
            do
                MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
                RESULT_DIR="$EXTERNAL_DIR/results/gamo/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                # model_path, result_path, cost_function, imagePerPerson
                train_gpu $MODEL $RESULT_DIR $i 30
                continue_train $MODEL $RESULT_DIR $i 30 50 51 50
            done

        done
    done
done

