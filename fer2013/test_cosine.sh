#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 500 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/training -outDir $RESULT_DIR/rep-$1/train -imgDim 48 -channelSize 1 -cuda

        ../batch-represent/main.lua -batchSize 500 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/privatetest -outDir $RESULT_DIR/rep-$1/test -imgDim 48 -channelSize 1 -cuda
   fi

   if [ -d $RESULT_DIR/rep-$1/train ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test
   fi
}
for i in 0.25 0.5 
do
    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    do
    
        RESULT_DIR="$WORK_DIR/data/results_cosine/nn4.small2/alpha$i"
    
        test $j

    done
    
    RESULT_DIR="$WORK_DIR/data/results_cosine/nn4.small2/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "FER_nn4.small2_$i"
    fi

done