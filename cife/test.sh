#!/usr/bin/env bash

imgDim=64
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
AUGMENTED_DIR="augmented${imgDim}"
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"


test ()
{
   echo $RESULT_DIR/rep-$1
   if [ -d $RESULT_DIR/rep-$1/test ] && [ ! -f $RESULT_DIR/rep-$1/test/accuracies_${2}.txt ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName ${DATA_LABEL} --counter $j --alg $2
   fi

   if [ -d $RESULT_DIR/rep-$1/train ] && [ ! -f $RESULT_DIR/rep-$1/train/svm_cife_confusion.png ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName ${DATA_LABEL} --train 1 --counter $j --alg $2
   fi
}

cd ../training



for alg in  nn svm #knn nn rf poly
do
    for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR # $ALIGNED_DIR
    do
        for DATA_LABEL in  cife # gamo fer2013
        do
            LABELED_DATA_DIR="$PWD/../${DATA_LABEL}/data/$DATA_DIR"
            for embSize in 128
            do
                for MODEL_NAME in vgg-face alexnet nn4
                do
                    for i in t_entropy multi t_global margin crossentropy s_cosine t_orj dist_ratio kldiv lmnn s_double_margin t_improved s_hadsell s_double_margin lmnn softPN lsss
                    do
                        for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
                        #for j in 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100

                        do
                            RESULT_DIR="$EXTERNAL_DIR/results/cife/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"
                            test $j  $alg

                        done
                    done
                done
            done
        done
    done
done

