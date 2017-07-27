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
            filename = os.path.join(input_dir, file)
            print filename
            if filename.endswith('.mat'):
                datas = scipy.io.loadmat(filename, mdict=None, appendmat=True)
                all_data = []
                for data in datas['trainImage']:
                    all_data.append(data[0])
                last_data = np.array(all_data)
                ds = datas['result'][1]
                labels = datas['result'][3]
                for i, d in enumerate(zip(ds, labels)):
                    img = Image.fromarray((d[0]).astype(np.uint8), 'RGB')
                    directory = os.path.join(output_dir, str(d[1][0][0]))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, '%s%s%s' % (file.split('.')[0], str(i), datas['result'][0][i][0]))
                    img.save(path, "JPEG", quality=100, optimize=True)


if __name__ == '__main__':
    mat2image("/home/cenk/Documents/openface-v2/disfa/data/Desktop", "/Users/cenk/Desktop/bau/openface/disfa/output")
