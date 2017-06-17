#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"



#ALIGNED_DIR="$PWD/data/aligned64"
#
##PREPROCESS
#if [ ! -d $ALIGNED_DIR/train ]; then
#    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/
#    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
#fi
#
#
#ALIGNED_DIR="$PWD/data/aligned48"
#
##PREPROCESS
#if [ ! -d $ALIGNED_DIR/train ]; then
#    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 48 --rgb 0 --fallbackLfw $RAW_DIR/
#    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
#fi

ALIGNED_DIR="$PWD/data/notaligned64"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 64 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
fi
#
#ALIGNED_DIR="$PWD/data/notaligned48"
#
##PREPROCESS
#if [ ! -d $ALIGNED_DIR/train ]; then
#    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 48 --rgb 0 --fallbackLfw $RAW_DIR/ --aligned 0
#    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.3
#fi


ALIGNED_DIR="$PWD/data/data_005_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.95
fi

ALIGNED_DIR="$PWD/data/data_010_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.90
fi

ALIGNED_DIR="$PWD/data/data_015_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.85
fi

ALIGNED_DIR="$PWD/data/data_020_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.80
fi

ALIGNED_DIR="$PWD/data/data_025_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.75
fi

ALIGNED_DIR="$PWD/data/data_030_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.70
fi


ALIGNED_DIR="$PWD/data/data_040_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.60
fi

ALIGNED_DIR="$PWD/data/data_050_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.50
fi

ALIGNED_DIR="$PWD/data/data_060_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.40
fi

ALIGNED_DIR="$PWD/data/data_070_96"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size 96 --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.30
fi












