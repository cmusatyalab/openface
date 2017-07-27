#!/usr/bin/env bash

MODEL_PATH=/home/cenk/Documents/openface-v2/models/mine

for j in 28 32 64
do
    for i in alexnet.def.lua nn4.def.lua vgg-face.def.lua
    do
        th util/print-network-table.lua -modelDef $MODEL_PATH/$j/$i -imgDim $j
    done
done