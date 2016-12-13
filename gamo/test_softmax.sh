#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 250 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -channelSize 3 -removeLast 1

        ../batch-represent/main.lua -batchSize 250 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -channelSize 3 -removeLast 1

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test

   fi
}



for j in {1..250}
do

    RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small3/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small2/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small1/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/nn2/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/nn4/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/vgg/"

    test $j

    RESULT_DIR="$WORK_DIR/data/results_softmax/vgg.small1/"

    test $j
done

RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small3/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small3_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small2/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small2_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/nn4.small1/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4.small1_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/nn2/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn2_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/nn4/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_nn4_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/vgg/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg_$i"
fi

RESULT_DIR="$WORK_DIR/data/results_softmax/vgg.small1/"
if [ -d $RESULT_DIR ];then
    python ../util/create_table.py --workDir $RESULT_DIR --title "GAMO_vgg.small1_$i"
fi

