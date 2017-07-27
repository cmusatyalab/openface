# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 01/12/2016 """
import math

import os
from PIL import Image
from PIL import ImageFilter

__author__ = 'cenk'


def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, expand=False):
    if center is None:
        return image.rotate(angle)
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def augment(input_dir, output):
    for _, dirs, _ in os.walk(input_dir):
        for dir in dirs:
            for _, subdir, files in os.walk(input_dir + '/' + dir):
                for file in files[:1]:
                    filename = input_dir + '/' + dir + '/' + file
                    output_file = output + '/' + dir
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)
                    im = Image.open(filename)
                    gaussian_filter = ImageFilter.GaussianBlur(radius=0.01)
                    unsharp_filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                    median_filter = ImageFilter.MedianFilter(size=3)
                    mode_filter = ImageFilter.ModeFilter(size=3)
                    for i, filt in enumerate(
                            [gaussian_filter,
                             unsharp_filter,
                             median_filter,
                             mode_filter
                             ]):
                        for angle in [0, 10, 20, 30]:
                            scale_rotate_translate(im.filter(filt), angle=angle).save(
                                '%s/%s%s.%s.jpg' % (output_file, '.'.join(file.split('.')[:-1]), angle, i * 10))
                            scale_rotate_translate(im.filter(filt).transpose(Image.FLIP_LEFT_RIGHT), angle=angle).save(
                                '%s/%s%s%s.jpg' % (output_file, '.'.join(file.split('.')[:-1]), angle, i))


if __name__ == '__main__':
    input_dir = '/Users/cenk/Desktop/bau/openface/cife/data/raw/train'
    output = '/Users/cenk/Desktop/bau/openface/cife/data/augment/train'
    augment(input_dir, output)
