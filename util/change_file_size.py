# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """

from random import shuffle

import numpy as np
import os

import PIL
from PIL import Image

__author__ = 'cenk'


def change_file_size(input_dir):
    pairs_same, pairs_else, names = [], [], []
    for _, dirs, _ in os.walk(input_dir):
        for dir in dirs:
            subdirs = os.path.join(input_dir, dir)
            for _, subdir, files in os.walk(subdirs):
                for file in files:
                    if '.DS_Store' not in file:
                        try:
                            file_name = os.path.join(subdirs, file)
                            img = Image.open(file_name)
                            img = img.resize((64, 64))
                            print file_name
                            img.save(file_name)
                        except Exception as e:
                            print e.message


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputDir', type=str, help='an integer for the accumulator')

    args = parser.parse_args()
    change_file_size(args.inputDir)
