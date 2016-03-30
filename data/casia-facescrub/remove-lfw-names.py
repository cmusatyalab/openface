#!/usr/bin/env python2

import os
import shutil
import sys

lfwDir = '../lfw/raw'
lfwNames = os.listdir(lfwDir)
lfwNames = [name.replace("_", "").lower() for name in lfwNames]

names = os.listdir('raw')

def inLfw(name):
    name = name.strip().lower()
    for lfwName in lfwNames:
        if lfwName == name:
            # print("(lfwName, name): ({}, {})".format(lfwName, name))
            return True
    return False

for name in names:
    if inLfw(name):
        print('Deleting: {}'.format(name))
        shutil.rmtree(os.path.join('raw', name))
