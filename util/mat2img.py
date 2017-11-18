# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 02/02/2017 """
import numpy as np
import os
import scipy.io
from PIL import Image

__author__ = 'cenk'


def mat2image(input_dir, output_dir):
    for _, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith('i'):
                filename = os.path.join(input_dir, file)
                datas = scipy.io.loadmat(filename, mdict=None, appendmat=True)
                labels = scipy.io.loadmat("/home/cenk/Documents/openface-v2/disfa/data/au1.mat", mdict=None,
                                          appendmat=True)
                ds = datas['images']
                labels = np.apply_along_axis(lambda x: "0" if x[0] == 1 else "1", 1, labels['au1'])

                for i, d in enumerate(zip(ds, labels)):
                    img = Image.fromarray(d[0][0])
                    directory = os.path.join(output_dir, str(d[1]))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, '%s.jpg' % (str(i)))
                    img.save(path, "JPEG", quality=100, optimize=True)


if __name__ == '__main__':
    mat2image("/home/cenk/Documents/openface-v2/disfa/data", "/home/cenk/Documents/openface-v2/disfa/data/au1")
