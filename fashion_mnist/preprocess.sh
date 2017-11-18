#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
RESULT_DIR="$PWD/data/results"

if [ ! -d $RAW_DIR ]; then
    python ../util/byte_to_image.py --inputDir $WORK_DIR/data/ --outputDir $WORK_DIR/data/raw
fi