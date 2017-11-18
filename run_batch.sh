#!/usr/bin/env bash

HERE=$PWD

echo "Start"
cd $HERE/gamo
sh create_batch.sh


cd $HERE/cife
sh create_batch.sh


cd $HERE/fer2013
sh create_batch.sh


cd $HERE/cifar10
sh create_batch.sh


cd $HERE/mnist
sh create_batch.sh

