#!/usr/bin/env bash

imgDim=32
WORK_DIR=$PWD
EXTERNAL_DIR="/media/cenk/DISK_5TB/losses"
NOT_ALIGNED_DIR="raw"
ALIGNED_DIR="raw"


create_batch ()
{
    echo $RESULT_DIR/model_$1.t7
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -f $RESULT_DIR/rep-$1/test/reps.csv ]; then

        ../batch-represent/main.lua -batchSize 50 -model $RESULT_DIR/model_$1.t7 -cuda   \
            -data $LABELED_DATA_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim $imgDim -channelSize 3 $2 -cache
    fi

    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -f $RESULT_DIR/rep-$1/train/reps.csv ]; then

        ../batch-represent/main.lua -batchSize 50 -model $RESULT_DIR/model_$1.t7  -cuda \
            -data $LABELED_DATA_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim $imgDim -channelSize 3 $2 -cache
    fi
}

cd ../training

for DATA_DIR in $NOT_ALIGNED_DIR #$AUGMENTED_DIR $ALIGNED_DIR
do
    for DATA_LABEL in cifar10
    do
        LABELED_DATA_DIR="$PWD/../${DATA_LABEL}/data/$DATA_DIR"
        for embSize in 128
        do
            for MODEL_NAME in alexnet vgg-face nn4
            do
                for i in t_entropy multi margin crossentropy s_cosine t_orj dist_ratio t_improved s_hadsell s_double_margin lmnn softPN lsss histogram  t_global
                do
                    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
                    do
                        RESULT_DIR="$EXTERNAL_DIR/results/cifar10/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

                        create_batch $j "-removeLast 0"
                    done
                done
                for i in kldiv s_hinge
                do
                    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
                    do
                        RESULT_DIR="$EXTERNAL_DIR/results/cifar10/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

                        create_batch $j "-removeLast 1"
                    done
                done
            done
        done
    done
done

