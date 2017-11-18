#!/usr/bin/env bash

HERE=$PWD

echo "Start"
cd $HERE/gamo

sh train.sh
sh create_batch.sh
#sh test.sh

cd $HERE/cife

sh train.sh
sh create_batch.sh
#sh test.sh

cd $HERE/cifar10

sh train.sh
sh create_batch.sh
#sh test.sh

cd $HERE/mnist

sh train.sh
sh create_batch.sh
#sh test.sh




