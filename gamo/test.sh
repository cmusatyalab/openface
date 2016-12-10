#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -channelSize 3

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -channelSize 3

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test

   fi
}



for i in 0.1 0.2 0.3 0.4 0.5
do
    for j in {1..30}
    do

        RESULT_DIR="$WORK_DIR/data/results/nn4.small3/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/nn4.small2/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/nn4.small1/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/nn2/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/nn4/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/vgg/alpha$i"

        test $j

        RESULT_DIR="$WORK_DIR/data/results/vgg.small1/alpha$i"

        test $j
     done

     RESULT_DIR="$WORK_DIR/data/results/nn4.small3/alpha$i"
     if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "database=GAMO network=nn4.small3 alpha=$i"
     fi

     RESULT_DIR="$WORK_DIR/data/results/nn4.small2/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=nn4.small2 alpha=$i'
     fi

     RESULT_DIR="$WORK_DIR/data/results/nn4.small1/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=nn4.small1 alpha=$i'
     fi

     RESULT_DIR="$WORK_DIR/data/results/nn2/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=nn2 alpha=$i'
     fi

     RESULT_DIR="$WORK_DIR/data/results/nn4/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=nn4 alpha=$i'
     fi

     RESULT_DIR="$WORK_DIR/data/results/vgg/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=vgg alpha=$i'
     fi

     RESULT_DIR="$WORK_DIR/data/results/vgg.small1/alpha$i"
     if [ -d $RESULT_DIR ];then
         python ../util/create_table.py --workDir $RESULT_DIR --title 'database=GAMO network=vgg.small1 alpha=$i'
     fi

done