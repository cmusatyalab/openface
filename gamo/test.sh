#!/bin/bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/test ]; then

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -channelSize 3 $2
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
            --testDir $RESULT_DIR/rep-$1/test
   fi



}

test_train_dataset()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -channelSize 3 $2
         python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
            --testDir $RESULT_DIR/rep-$1/test --train 1
    fi


}

for i in loglikelihood cosine l1hinge triplet
do
    for j in {30..50}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"
        if [ $i == "loglikelihood" ]; then
            echo test $j "-removeLast 1"
            test $j "-removeLast 1"

        else
            echo test $j "-removeLast 0"
            test $j "-removeLast 0"
        fi
    done

    for j in {30..50}
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"
        if [ $i == "loglikelihood" ]; then
            echo test_train_dataset $j "-removeLast 1"
            test_train_dataset $j "-removeLast 1"
        else
            echo test_train_dataset $j "-removeLast 0"
            test_train_dataset $j "-removeLast 0"
        fi
    done

   RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ -f $RESULT_DIR/test_score.log  ] && [ -f $RESULT_DIR/train_score.log ];then

        python ../util/create_table.py --workDir $RESULT_DIR --title "CIFE_nn4.small2"
    fi

done


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
