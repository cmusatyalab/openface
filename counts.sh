#!/usr/bin/env bash

j=10
if [ $(($j % 5))  -eq  0 ];
then
    echo "Waiting"
    wait
fi