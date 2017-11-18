# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """
import argparse

import os
import re
import pandas as pd

__author__ = 'cenk'

paths = {}


def check_last_train(path):
    for _, folders, files in os.walk(path):
        if files and ('nn4' in path or 'alexnet' in path or 'vgg-face' in path):
            for f in files:
                if f.endswith('.t7'):
                    if 'model' in f:
                        f = f.replace('model_', '').replace('.t7', '')
                        if path in paths:
                            paths[path].append(int(f))
                        else:
                            paths[path] = [int(f)]
        if folders:
            for folder in folders:
                absfolder = os.path.join(path, folder)
                check_last_train(absfolder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    check_last_train(args.path)
    for path, numbers in paths.iteritems():
        print path, max(numbers)
