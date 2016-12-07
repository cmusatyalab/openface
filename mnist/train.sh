#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
RESULT_DIR="$PWD/data/results"

if [ ! -d $RAW_DIR ]; then
    python ../util/byte_to_image.py --inputDir . --outputDir $RAW_DIR --rgb 0
fi

cd ../training

th main.lua -data $RAW_DIR/train -modelDef $WORK_DIR/model/nn4.small2.def.lua -cache $WORK_DIR/data/cache  \
 -save $RESULT_DIR  -nDonkeys 2  -cuda -cudnn -peoplePerBatch 10 -imagesPerPerson 25 -testPy ../evaluation/classify.py \
 -testDir $RAW_DIR/test -testBatchSize 100 -epochSize 1000 -nEpochs 10 -imgDim 28 -channelSize 1
