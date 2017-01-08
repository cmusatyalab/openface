#!/usr/bin/env bash

while true
do
    echo "Start"
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
    echo "Finish"
    git add .
    git commit -am "auto commit"
    git push origin bau
    echo "Commited"
done