#!/usr/bin/env bash

HERE=$PWD

echo "Start"
cd $HERE/gamo

sh lsss_hist.sh

cd $HERE/cife

sh lsss_hist.sh

cd $HERE/cifar10

sh lsss_hist.sh

cd $HERE/mnist

sh lsss_hist.sh
