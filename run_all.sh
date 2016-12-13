#!/usr/bin/env bash

HERE=$PWD


for i in triplet softmax  l1_hinge cosine
do
    for j in gamo cife fer2013
    do
        cd $HERE/$j
        sh train_$i.sh

    done

    for j in gamo cife fer2013
    do
        cd $HERE/$j
        sh test_$i.sh

    done

done

