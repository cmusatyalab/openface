#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/training -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 80  -peoplePerBatch 7 -imagesPerPerson 50 -testPy ../evaluation/classify.py -testing \
            -testDir $ALIGNED_DIR/test -testBatchSize 100 -epochSize 250 -nEpochs 50 -imgDim 48 -channelSize 1 -criterion loglikelihood
    fi
}


#PREPROCESS
if [ ! -d $RAW_DIR ]; then
    python ../util/csv_to_image.py --inputFile fer2013.csv --outputDir $RAW_DIR
fi
if [ ! -d $ALIGNED_DIR/training ]; then
    python ../util/align-dlib.py $RAW_DIR/training align  outerEyesAndNose $ALIGNED_DIR/training --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/training
fi
if [ ! -d $ALIGNED_DIR/publictest ]; then
    python ../util/align-dlib.py $RAW_DIR/publictest align  outerEyesAndNose $ALIGNED_DIR/publictest --size 48 --rgb 0 \
        --fallbackLfw $RAW_DIR/publictest
fi

if [ ! -d $ALIGNED_DIR/privatetest ]; then
    python ../util/align-dlib.py $RAW_DIR/privatetest align  outerEyesAndNose $ALIGNED_DIR/privatetest --size 48 --rgb 0  \
        --fallbackLfw $RAW_DIR/privatetest
fi


cd ../training


MODEL=$WORK_DIR/../models/mine/nn4.small2.def.48_1.lua
RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small2"
if [ ! -d $RESULT_DIR ]; then

    train $MODEL $RESULT_DIR
fi
