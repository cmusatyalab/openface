#!/usr/bin/env python2
#
# Detect outlier faces (not of the same person) in a directory
# of aligned images.
# Brandon Amos
# 2016/02/14
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import glob

import numpy as np
np.set_printoptions(precision=2)

from sklearn.covariance import EllipticEnvelope
from sklearn.metrics.pairwise import euclidean_distances

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--threshold', type=int, default=0.9)
    parser.add_argument('directory')

    args = parser.parse_args()

    net = openface.TorchNeuralNet(args.networkModel, args.imgDim, cuda=args.cuda)

    reps = []
    paths = sorted(list(glob.glob(os.path.join(args.directory, '*.png'))))
    for imgPath in paths:
        reps.append(net.forwardPath(imgPath))

    mean = np.mean(reps, axis=0)
    dists = euclidean_distances(reps, mean)
    outliers = []
    for path, dist in zip(paths, dists):
        dist = dist.take(0)
        if dist > args.threshold:
            outliers.append((path, dist))

    print("Found {} outlier(s) from {} images.".format(len(outliers), len(paths)))
    for path, dist in outliers:
        print(" + {} ({:0.2f})".format(path, dist))

if __name__ == '__main__':
    main()
