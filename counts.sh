#!/usr/bin/env bash


for j in test train
do
    for i in angry disgust fear happy neutral sad surprise suprise
    do
        echo $1 $j $i
        ls $1/data/aligned/$j/$i | wc -l
    done
done