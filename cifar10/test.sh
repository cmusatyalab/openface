#!/bin/bash

WORK_DIR=$PWD
ALIGNED_DIR="$PWD/data/raw"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 32 $2

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 32 $2
         python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test
   fi


}

for i in triplet siamese
do
    for j in {1..100}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

            test $j "-removeLast 0"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFAR10_nn4.small2"
    fi

done


for i in contrastive
do
    for j in {1..100}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"
        test $j "-removeLast 1"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFAR10_nn4.small2"
    fi

done

