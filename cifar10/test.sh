#!/bin/bash
imgDim=32
WORK_DIR=$PWD
ALIGNED_DIR="data/raw"
DATA_DIR="raw"
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"

test_cifar10()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then

        ../batch-represent/main.lua -batchSize 10 -model $RESULT_DIR/model_$1.t7  \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim $imgDim -channelSize 3 $2

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName cifar --counter $j
        rm -rf $RESULT_DIR/model_$1.t7
        rm -rf $RESULT_DIR/optimState_$1.t7
        rm -rf $RESULT_DIR/rep-$1
   fi
}

while true
do
    for DATA_DIR in  $ALIGNED_DIR
    do
        for embSize in 128
        do
            for MODEL_NAME in  toynet alexnet vgg-face
            do
                for i in crossentropy s_cosine t_orj dist_ratio
                do
                    for j in {1..250}
                    do
                        RESULT_DIR="$EXTERNAL_DIR/results/cifar10/raw_${embSize}/${i}/$MODEL_NAME"

                        test_cifar10 $j "-removeLast 0"
                    done
                done
                for i in kldiv s_hinge
                do
                    for j in {1..250}
                    do
                        RESULT_DIR="$EXTERNAL_DIR/results/cifar10/raw_${embSize}/${i}/$MODEL_NAME"

                        test_cifar10 $j "-removeLast 1"
                    done
                done
            done
        done

    done
done