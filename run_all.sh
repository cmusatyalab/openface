#!/usr/bin/env bash

HERE=$PWD

cd $HERE/gamo

sh train.sh

cd $HERE/cife

sh train.sh

cd $HERE/cifar10

sh train.sh

cd $HERE/fer2013

sh train.sh

cd $HERE/mnist

sh train.sh