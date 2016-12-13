#!/usr/bin/env bash

HERE=$PWD

cd $HERE/cife
sh train_triplet.sh

cd $HERE/fer2013
sh train_triplet.sh

cd $HERE/gamo
sh train_triplet.sh


cd $HERE/cife
sh test_triplet.sh

cd $HERE/fer2013
sh test_triplet.sh

cd $HERE/gamo
sh test_triplet.sh


cd $HERE/cife
sh train_softmax.sh

cd $HERE/fer2013
sh train_softmax.sh

cd $HERE/gamo
sh train_softmax.sh


cd $HERE/cife
sh test_softmax.sh

cd $HERE/fer2013
sh test_softmax.sh

cd $HERE/gamo
sh test_softmax.sh



cd $HERE/cife
sh train_l1_hinge.sh

cd $HERE/fer2013
sh train_l1_hinge.sh

cd $HERE/gamo
sh train_l1_hinge.sh


cd $HERE/cife
sh test_l1_hinge.sh

cd $HERE/fer2013
sh test_l1_hinge.sh

cd $HERE/gamo
sh test_l1_hinge.sh






cd $HERE/cife
sh train_cosine.sh

cd $HERE/fer2013
sh train_cosine.sh

cd $HERE/gamo
sh train_cosine.sh


cd $HERE/cife
sh test_cosine.sh

cd $HERE/fer2013
sh test_cosine.sh

cd $HERE/gamo
sh test_cosine.sh


