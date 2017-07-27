# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """

from random import shuffle

import numpy as np
import os

__author__ = 'cenk'

test_size = 10000


def merge_pairs(pairs_same, pairs_else):
    pairs = []
    if len(pairs_same) > len(pairs_else):
        pairs.extend(pairs_else)
        shuffle(pairs_same)
        pairs.extend(pairs_same[:len(pairs_else)])
    elif len(pairs_else) > len(pairs_same):
        pairs.extend(pairs_same)
        shuffle(pairs_else)
        pairs.extend(pairs_else[:len(pairs_same)])
    else:
        pairs.extend(pairs_same)
        pairs.extend(pairs_else)
    print "same size:%s diff size:%s" % (len(pairs_same), len(pairs_else))
    return pairs


def save(pairs, output_path):
    shuffle(pairs)
    with open(output_path, 'wb') as f:
        for item in pairs:
            f.write("%s\n" % item)


def create_pairs(args):
    global test_size
    input_dir = args.inputDir
    output_file = args.outputFile
    pairs_same, pairs_else, names = [], [], []
    for _, dirs, _ in os.walk(input_dir):
        for dir in dirs:
            for _, subdir, files in os.walk(input_dir + '/' + dir):
                for file in files:
                    names.append(file)

    while len(pairs_else) != test_size / 2 or len(pairs_same) != test_size / 2:
        val1, val2 = np.random.choice(np.array(names), 2)
        name1, n1, _ = val1.split('.')
        name2, n2, _ = val2.split('.')
        if n1 == 'png' or n2 == 'png':
            continue
        if name1 == name2:
            pair = '%s %s %s' % (name1, n1, n2)
            if len(pairs_same) != test_size / 2:
                pairs_same.append(pair)
        else:
            pair = '%s %s %s %s' % (name1, n1, name2, n2)
            if len(pairs_else) != test_size / 2:
                pairs_else.append(pair)
    pairs = merge_pairs(pairs_same, pairs_else)
    save(pairs, output_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--outputFile', type=str, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    create_pairs(args)
