#!/bin/bash
imgDim=32
WORK_DIR=$PWD
ALIGNED_DIR="data/raw"
DATA_DIR="raw"

test()
{
    if  [  -d $RESULT_DIR/rep-$1/train ]; then

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName shn --counter $j
        rm -rf $RESULT_DIR/model_$1.t7
        rm -rf $RESULT_DIR/optimState_$1.t7
   fi
}


for embSize in 128
do
    for MODEL_NAME in  toynet alexnet vgg-face
    do
        for i in crossentropy s_cosine t_orj dist_ratio
        do
            for j in {1..250}
            do
                RESULT_DIR="$WORK_DIR/results/raw_${embSize}/${i}/$MODEL_NAME"

                test_cifar10 $j "-removeLast 0"
            done
        done
        for i in kldiv s_hinge
        do
            for j in {1..250}
            do
                RESULT_DIR="$WORK_DIR/results/raw_${embSize}/${i}/$MODEL_NAME"

                test_cifar10 $j "-removeLast 1"
            done
        done
    done
done
