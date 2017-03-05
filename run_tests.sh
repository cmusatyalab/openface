#!/usr/bin/env bash

HERE=$PWD

echo "Start"
cd $HERE/gamo
sh test.sh

cd $HERE/cife
sh test.sh

cd $HERE/fer2013
sh test.sh

cd $HERE/cifar10
sh test.sh

cd $HERE/mnist
sh test.sh
