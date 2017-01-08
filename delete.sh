#!/bin/bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


delete_model ()
{
    if [ -f $RESULT_DIR/model_$1.t7  ] || [ -f $RESULT_DIR/optimState_$1.t7  ]; then
        echo  $RESULT_DIR/model_$1.t7
        rm -rf  $RESULT_DIR/model_$1.t7
        rm -rf  $RESULT_DIR/optimState_$1.t7
   fi

}
while true
do
for m in  gamo cife fer2013 cifar10
do
    for i in cosine l2loss triplet loglikelihood
    do
        for j in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
        do
            RESULT_DIR="$m/data/results_$i/nn4.small2"

             delete_model $j

        done

    done
done
sleep 120
done