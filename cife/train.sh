#!/usr/bin/env bash


sh train_triplet.sh
sh train_softmax.sh
sh train_cosine.sh
sh train_l1_hinge.sh

sh test_triplet.sh
sh test_softmax.sh
sh test_cosine.sh
sh test_l1_hinge.sh