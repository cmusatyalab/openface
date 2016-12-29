#!/usr/bin/env bash

HERE=$PWD



cd $HERE/cife
sh train.sh

cd $HERE/training/
th main.lua -data /home/bau_fbe/Documents/openface-v2/gamo/data/aligned/train \
    -modelDef /home/bau_fbe/Documents/openface-v2/models/mine/nn4.small2.def.64_1.lua \
    -cache /home/bau_fbe/Documents/openface-v2/gamo/data/cache \
    -save /home/bau_fbe/Documents/openface-v2/gamo/data/results_kl/nn4.small2/ \
     -nDonkeys 2  -peoplePerBatch 7 -imagesPerPerson 25 -testing -epochSize 500 -nEpochs 245 -imgDim 64 \
     -criterion triplet -retrain /home/bau_fbe/Documents/openface-v2/gamo/data/results_kl/nn4.small2/model_5.t7  -epochNumber 6 -cuda

cd $HERE/gamo
sh train.sh

cd $HERE/fer2013
sh train.sh


cd $HERE/cifar10
sh train.sh

cd $HERE/gamo
sh test.sh

cd $HERE/cife
sh test.sh

cd $HERE/fer2013
sh test.sh

cd $HERE/cifar10
sh test.sh