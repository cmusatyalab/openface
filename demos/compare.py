#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015 Carnegie Mellon University
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

import numpy as np
np.set_printoptions(precision=2)

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import facenet
import facenet.helper
from facenet.data import iterImgs

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
facenetModelDir = os.path.join(modelDir, 'facenet')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--dlibRoot', type=str,
                    default=os.path.expanduser("~/src/dlib-18.16/python_examples"),
                    help="dlib directory with the dlib.so Python library.")
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(facenetModelDir, 'nn4.v1.t7'))
parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

sys.path.append(args.dlibRoot)
import dlib

from facenet.alignment import NaiveDlib # Depends on dlib.
if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(time.time()-start))

start = time.time()
align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = facenet.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
if args.verbose:
    print("Loading the dlib and FaceNet models took {} seconds.".format(time.time()-start))

def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    img = cv2.imread(imgPath)
    if img is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    if args.verbose:
        print("  + Original size: {}".format(img.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time()-start))

    start = time.time()
    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time()-start))

    start = time.time()
    rep = net.forwardImage(alignedFace)
    if args.verbose:
        print("  + FaceNet forward pass took {} seconds.".format(time.time()-start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

for (img1, img2) in itertools.combinations(args.imgs, 2):
    d = getRep(img1) - getRep(img2)
    print("Comparing {} with {}.".format(img1, img2))
    print("  + Squared l2 distance between representations: {}".format(np.dot(d, d)))
