#!/usr/bin/env bash

HERE=$PWD

cd $HERE/mnist
sh train.sh


cd $HERE/cifar10
sh train.sh



