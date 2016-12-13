#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 500 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -channelSize 3 -cuda

        ../batch-represent/main.lua -batchSize 500 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -channelSize 3 -cuda

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test

   fi
}

for i in 0.25 0.5 
do
    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
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
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn4.small3_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small2/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn4.small2_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4.small1/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn4.small1_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn2/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn2_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/nn4/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn4_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_vgg_$i"
    fi
    
    RESULT_DIR="$WORK_DIR/data/results_l1_hinge/vgg.small1/alpha$i"
    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_vgg.small1_$i"
    fi
done