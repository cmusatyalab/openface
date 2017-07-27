#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
RESULT_DIR="$PWD/data/results"

if [ ! -d $RAW_DIR ]; then
    python utils.py
fi