# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """

import os

__author__ = 'cenk'


def change_filename(args):
    path = args.path
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            for _, subdir, files in os.walk(path + '/' + dir):
                counter = 0
                for file in files:
                    counter += 1
                    splitted = file.split('.')
                    splitted[1] = ("%d" % counter).zfill(4)
                    name = '.'.join(splitted)
                    os.rename(path + '/' + dir + '/' + file, path + '/' + dir + '/' + name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    args = parser.parse_args()
    change_filename(args)
