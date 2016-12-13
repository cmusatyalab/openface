#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -channelSize 3 -cuda

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -channelSize 3 -cuda

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test

   fi
}

for i in 0.25 0.5 
do
    for j in 1 10 20 30 40 50 60 70 80 90 100
    do
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small3/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small2/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small1/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn2/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg/alpha$i"
    
        test $j
    
        RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg.small1/alpha$i"
    
        test $j
    done
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small3/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small3_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small2/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small2_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small1/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small1_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn2/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn2_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg.small1/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg.small1_$i"
    fi
done