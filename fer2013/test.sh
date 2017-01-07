#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 10 -model $RESULT_DIR/model_$1.t7 -cuda  \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -imgDim 48 -channelSize 1 $2 -cuda

        ../batch-represent/main.lua -batchSize 10 -model $RESULT_DIR/model_$1.t7 -cuda \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -imgDim 48 -channelSize 1 $2 -cuda

        if [ "$1" -lt 1500 ]; then
            rm -rf  $RESULT_DIR/model_$1.t7
            rm -rf  $RESULT_DIR/optimState_$1.t7
        else
            mkdir -p /media/cenk/DISK500GB/fer2013/$3/
            mv $RESULT_DIR/model_$1.t7 /media/cenk/DISK500GB/fer2013/$3/
            mv $RESULT_DIR/optimState_$1.t7 /media/cenk/DISK500GB/fer2013/$3/
        fi
    fi

   if [ -d $RESULT_DIR/rep-$1/train ]; then

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test
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
        python ../util/create_table.py --workDir $RESULT_DIR --title "FER2013_nn4.small2"
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
        python ../util/create_table.py --workDir $RESULT_DIR --title "FER2013_nn4.small2"
    fi

done
