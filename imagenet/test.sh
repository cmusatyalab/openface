#!/usr/bin/env bash

imgDim=96
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"




test ()
{
   if [ -d $RESULT_DIR/rep-$1/test ] && [ ! -f $RESULT_DIR/rep-$1/test/accuracies_${2}.txt ]; then
        echo $RESULT_DIR/rep-$1 test $alg
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName ${DATA_LABEL} --counter $j --alg $2
   fi

   if [ -d $RESULT_DIR/rep-$1/train ] && [ ! -f $RESULT_DIR/rep-$1/train/$2_imagenet_confusion.png ]; then
        echo $RESULT_DIR/rep-$1 train $alg
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
                --testDir $RESULT_DIR/rep-$1/test --pathName ${DATA_LABEL} --train 1 --counter $j --alg $2
   fi
}

cd ../training



for alg in nn
do
    for DATA_LABEL in imagenet
    do
        LABELED_DATA_DIR="$PWD/../${DATA_LABEL}/data/"
        for embSize in 128
        do
            for MODEL_NAME in vgg-face  alexnet nn4
            do
                for i in margin crossentropy s_cosine t_orj dist_ratio kldiv lmnn s_double_margin t_improved s_hadsell s_double_margin lmnn softPN lsss s_hinge histogram t_global
                do
                    for j in  160 350 360
                    do
                        RESULT_DIR="$EXTERNAL_DIR/results/${DATA_LABEL}/${embSize}/$i/$MODEL_NAME"
                        test $j  $alg
                    done
                done
            done
        done
    done

done


