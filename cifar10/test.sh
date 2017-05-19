#!/bin/bash
imgDim=32
WORK_DIR=$PWD
ALIGNED_DIR="data/raw"
DATA_DIR="raw"
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"

test_cifar10()
{
   echo $RESULT_DIR/rep-$1
   if [ -d $RESULT_DIR/rep-$1/test ] && [ ! -f $RESULT_DIR/rep-$1/test/accuracies_${2}.txt ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName cifar --counter $j --alg $2
   fi

   if [ -d $RESULT_DIR/rep-$1/train ] && [ ! -f $RESULT_DIR/rep-$1/train/svm_cifar_confusion.png ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName cifar --train 1 --counter $j --alg $2
   fi
}

for alg in  svm #knn nn rf poly
do
    for embSize in 128
    do
        for MODEL_NAME in  "vgg-face"  "alexnet"  "nn4"
        do
            for i in t_entropy multi margin crossentropy s_cosine t_orj dist_ratio kldiv lmnn s_double_margin t_improved s_hadsell s_double_margin lmnn softPN lsss s_hinge
            do
                for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
                do
                    RESULT_DIR="$EXTERNAL_DIR/results/cifar10/raw_${embSize}/${i}/$MODEL_NAME"
                    test_cifar10 $j $alg
                done
            done
        done
    done
done