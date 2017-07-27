# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """
import csv

import numpy as np
import os
from PIL import Image

__author__ = 'cenk'

EMOTION_MAP = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
}


def csv_to_image(args):
    input_file = args.inputFile
    output_dir = args.outputDir
    with open(input_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', )
        header = next(spamreader)
        emotion_counter = {}
        counter = 0
        for row in spamreader:
            counter += 1
            emotion = EMOTION_MAP[int(row[0])]
            pixels = row[1].split(" ")
            usage = row[2].lower()
            if not usage in emotion_counter:
                emotion_counter[usage] = {}
            emotion_counter[usage][emotion] = emotion_counter[usage].get(emotion, 0) + 1
            pixels_arr = np.array(pixels, dtype=int).reshape(48, 48)
            img = Image.fromarray((pixels_arr).astype(np.uint8), 'L')
            directory = os.path.join(output_dir, os.path.join(usage, emotion))
            if not os.path.exists(directory):
                os.makedirs(directory)
            path = os.path.join(directory, '%s.%06d.jpg' % (emotion, emotion_counter[usage][emotion]))
            img.save(path, "JPEG", quality=100, optimize=True)
        print counter


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputFile', type=str, help='an integer for the accumulator')
    parser.add_argument('--outputDir', type=str, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    csv_to_image(args)
