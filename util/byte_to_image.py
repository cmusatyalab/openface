# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """
import struct

import numpy as np
import os
from PIL import Image
from numpy import int8, uint8

__author__ = 'cenk'


def read_and_save(img_file, lbl_file, datatype, output_dir, rgb=True):
    with open(lbl_file, 'rb') as file:
        magic_nr, size = struct.unpack(">II", file.read(8))
        lbl = np.fromfile(file, dtype=np.int8)

    with open(img_file, 'rb') as file:
        magic_nr, size, rows, cols = struct.unpack(">IIII", file.read(16))
        img = np.fromfile(file, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in xrange(len(lbl)):
        label, img1 = get_img(i)
        label = str(label)
        image = Image.fromarray(img1.astype(np.uint8))
        if rgb:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        directory = os.path.join(output_dir, os.path.join(datatype, label))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, '%s.%06d.jpg' % (label, i))
        image.save(path, "JPEG", quality=100, optimize=True)


def ubyte_to_image(args):
    input_dir = args.inputDir
    output_dir = args.outputDir

    train_img = os.path.join(input_dir, 'train-images.idx3-ubyte')
    train_lbl = os.path.join(input_dir, 'train-labels.idx1-ubyte')
    read_and_save(train_img, train_lbl, 'training', output_dir, rgb=args.rgb)

    test_img = os.path.join(input_dir, 't10k-images.idx3-ubyte')
    test_lbl = os.path.join(input_dir, 't10k-labels.idx1-ubyte')
    read_and_save(test_img, test_lbl, 'testing', output_dir, rgb=args.rgb)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--outputDir', type=str, help='sum the integers (default: find the max)')
    parser.add_argument('--rgb', type=int, default=1)

    args = parser.parse_args()
    ubyte_to_image(args)
