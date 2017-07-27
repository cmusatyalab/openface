# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """

import cPickle

import numpy as np
import os
from PIL import Image

__author__ = 'cenk'

counter = 1


def read_and_save(img_file, datatype, output_dir):
    global counter
    with open(img_file, 'rb') as file:
        dict = cPickle.load(file)
    labels = dict.get('fine_labels')
    data = dict.get('data')
    metafile = cPickle.load(open("data/cifar100/meta"))
    filenames = dict.get('filenames')
    for i, label in enumerate(labels):
        data_i = data[i]
        counter += 1
        label_name = metafile.get('fine_label_names')[label]
        data_i = data_i.reshape(3, 32, 32)
        data_i = np.transpose(data_i, (1, 2, 0))
        image = Image.fromarray(data_i.astype(np.uint8), 'RGB')

        directory = os.path.join(output_dir, os.path.join(datatype, label_name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = filenames[i]
        path = os.path.join(directory, '%s' % (filename))
        image.save(path, "JPEG", quality=100, optimize=True)


def to_image():
    output_dir = "data/raw/"
    datatype = "train"
    data = "data/cifar100/train"
    read_and_save(data, datatype, output_dir)
    datatype = 'test'
    data = "data/cifar100/test"
    read_and_save(data, datatype, output_dir)


if __name__ == '__main__':
    to_image()
