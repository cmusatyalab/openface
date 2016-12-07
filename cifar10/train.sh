#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
RESULT_DIR="$PWD/data/results"

if [ ! -d $RAW_DIR ]; then
    python utils.py
fi

cd ../training

th main.lua -data $RAW_DIR/train -modelDef $WORK_DIR/model/nn4.small2.def.lua -cache $WORK_DIR/data/cache  \
 -save $RESULT_DIR  -nDonkeys 10  -cuda -cudnn -peoplePerBatch 10 -imagesPerPerson 25 -testPy ../evaluation/classify.py \
 -testDir $RAW_DIR/test -testBatchSize 100 -epochSize 100 -nEpochs 50 -imgDim 32
