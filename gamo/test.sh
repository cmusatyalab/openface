#!/bin/bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"
CIFE_DIR="$PWD/../cife/data/aligned"

test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/gamo_train ]; then
        ../batch-represent/main.lua -batchSize 20 -model $RESULT_DIR/model_$1.t7  -cuda  \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/gamo_train -imgDim 64 -channelSize 3 $2

        ../batch-represent/main.lua -batchSize 20 -model $RESULT_DIR/model_$1.t7  -cuda \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/gamo_test -imgDim 64 -channelSize 3 $2

        ../batch-represent/main.lua -batchSize 20 -model $RESULT_DIR/model_$1.t7 -cuda  \
            -data $CIFE_DIR/train -outDir $RESULT_DIR/rep-$1/cife_train -imgDim 64 -channelSize 3 $2

        ../batch-represent/main.lua -batchSize 20 -model $RESULT_DIR/model_$1.t7 -cuda  \
            -data $CIFE_DIR/test -outDir $RESULT_DIR/rep-$1/cife_test -imgDim 64 -channelSize 3 $2

        mkdir -p /media/cenk/DISK500GB/gamo/$3/
        mv $RESULT_DIR/model_$1.t7 /media/cenk/DISK500GB/gamo/$3/
        mv $RESULT_DIR/optimState_$1.t7 /media/cenk/DISK500GB/gamo/$3/

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/cife_train \
        --testDir $RESULT_DIR/rep-$1/cife_test

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/gamo_train \
        --testDir $RESULT_DIR/rep-$1/gamo_test
   fi

}



for i in triplet siamese
do
    for j in  {1..5000}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

         test $j "-removeLast 0"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small2"
    fi

done
for i in contrastive
do
    for j in  {1..5000}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

         test $j "-removeLast 1"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small2"
    fi

done
