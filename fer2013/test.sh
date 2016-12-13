#!/bin/bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -imgDim 48 -channelSize 1

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -imgDim 48 -channelSize 1
   fi

   if [ -d $RESULT_DIR/rep-$1/train ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test
   fi
}


for i in loglikelihood cosine l1hinge triplet
do
    for j in {23..25}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

        test $j

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small2"
    fi

done


#
#
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small3/alpha$i"
#
#    test $j
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small1/alpha$i"
#
#    test $j
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn2/alpha$i"
#
#    test $j
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/nn4/alpha$i"
#
#    test $j
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/vgg/alpha$i"
#
#    test $j
#
#    RESULT_DIR="$WORK_DIR/data/results_triplet/vgg.small1/alpha$i"
#
#    test $j
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small3/alpha$i"
# if [ -d $RESULT_DIR ];then
#    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small3_$i"
# fi
#
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/nn4.small1/alpha$i"
# if [ -d $RESULT_DIR ];then
#     python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small1_$i"
# fi
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/nn2/alpha$i"
# if [ -d $RESULT_DIR ];then
#     python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn2_$i"
# fi
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/nn4/alpha$i"
# if [ -d $RESULT_DIR ];then
#     python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4_$i"
# fi
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/vgg/alpha$i"
# if [ -d $RESULT_DIR ];then
#     python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg_$i"
# fi
#
# RESULT_DIR="$WORK_DIR/data/results_triplet/vgg.small1/alpha$i"
# if [ -d $RESULT_DIR ];then
#     python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg.small1_$i"
# fi
