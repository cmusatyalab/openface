# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """

import cPickle

import numpy as np
import os
from PIL import Image

__author__ = 'cenk'

counter = 1
LABELS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
          9: 'truck'}


def read_and_save(img_file, datatype, output_dir):
    global counter
    with open(img_file, 'rb') as file:
        dict = cPickle.load(file)
    labels = dict.get('labels')
    data = dict.get('data')
    for i, label in enumerate(labels):
        data_i = data[i]
        counter += 1
        label_name = LABELS.get(label)
        data_i = data_i.reshape(3, 32, 32)
        data_i = np.transpose(data_i, (1, 2, 0))
        image = Image.fromarray(data_i.astype(np.uint8), 'RGB')

        directory = os.path.join(output_dir, os.path.join(datatype, label_name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, '%s.%06d.jpg' % (label_name, counter))
        image.save(path, "JPEG", quality=100, optimize=True)


def to_image():
    output_dir = "data/raw/"
    datatype = "training"
    for i in xrange(1, 5):
        data = "cifar-10-batches-py/data_batch_%s" % i
        read_and_save(data, datatype, output_dir)
    datatype = 'testing'
    data = "cifar-10-batches-py/test_batch"
    read_and_save(data, datatype, output_dir)


if __name__ == '__main__':
    to_image()
