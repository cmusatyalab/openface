#!/usr/bin/env bash

HERE=$PWD


cd $HERE/fer2013
sh train.sh

cd $HERE/cife
sh train.sh

cd $HERE/gamo
sh train.sh

cd $HERE/gamo
sh test.sh

cd $HERE/cife
sh test.sh

cd $HERE/fer2013
sh test.sh

cd $HERE/cifar10
sh train.sh

cd $HERE/cifar10
sh test.sh