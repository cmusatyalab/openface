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

imgdim=64
ALIGNED_DIR="$PWD/data/data_005_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.95
fi

ALIGNED_DIR="$PWD/data/data_010_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.90
fi

ALIGNED_DIR="$PWD/data/data_015_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.85
fi

ALIGNED_DIR="$PWD/data/data_020_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.80
fi

ALIGNED_DIR="$PWD/data/data_025_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.75
fi

ALIGNED_DIR="$PWD/data/data_030_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.70
fi


ALIGNED_DIR="$PWD/data/data_040_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.60
fi

ALIGNED_DIR="$PWD/data/data_050_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.50
fi

ALIGNED_DIR="$PWD/data/data_060_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.40
fi

ALIGNED_DIR="$PWD/data/data_070_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.30
fi


ALIGNED_DIR="$PWD/data/data_080_${imgdim}"

#PREPROCESS
if [ ! -d $ALIGNED_DIR/train ]; then
    python ../util/align-dlib.py $RAW_DIR/ align  outerEyesAndNose $ALIGNED_DIR/ --size ${imgdim} --fallbackLfw $RAW_DIR/  --aligned 0
    python ../util/create-train-val-split.py $ALIGNED_DIR --valRatio 0.20
fi











