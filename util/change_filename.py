# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """
import argparse

import os

__author__ = 'cenk'

mapping = {"A": "angry", "B": "disgust", "C": "fear", "D": "happy", "E": "neutral", "F": "sad", "G": "suprise"}


def change_filename(args):
    path = args.path
    for _, _, files in os.walk(path):
        for file in files:
            folder = mapping.get(file[0])

            name = "%s.%s" % (folder, file[1:])
            directory = os.path.join(path, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.rename(os.path.join(path, file), os.path.join(directory, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    args = parser.parse_args()
    change_filename(args)
