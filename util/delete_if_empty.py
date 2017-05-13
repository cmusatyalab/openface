# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """
import argparse

import os

__author__ = 'cenk'

def is_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) == 0


def delete_if_empty(path):
    counter = 0
    for _, folders, files in os.walk(path):
        if not files and not folders:
            print _
            os.rmdir(_)
        if folders:
            for folder in folders:
                absfolder = os.path.join(path, folder)
                delete_if_empty(absfolder)
        if files:
            for file in files:
                absfile = os.path.join(path, file)
                if is_zero_file(absfile):
                    print(absfile)
                    os.remove(absfile)
                    counter += 1

    if counter:
        print(counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    args = parser.parse_args()
    delete_if_empty(args.path)
