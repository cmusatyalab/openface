#!/usr/bin/env python2
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

import sys
sys.path.append(".")

import argparse
import numpy as np
import os
import random

import cv2

import openface
from openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('imgDir', type=str, help="Input image directory.")
parser.add_argument('--numImages', type=int, default=1000)
parser.add_argument('--model', type=str, help="TODO",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--outputFile', type=str,
                    help="Output file, stored in numpy serialized format.",
                    default="./unknown.npy")
parser.add_argument('--imgDim', type=int, help="Default image size.",
                    default=96)
args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.model, imgDim=args.imgDim, cuda=False)


def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        return None
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        return None

    alignedFace = align.align(args.imgDim, rgbImg, bb)
    if alignedFace is None:
        return None

    rep = net.forward(alignedFace)
    return rep

if __name__ == '__main__':
    allImgs = list(iterImgs(args.imgDir))
    imgObjs = random.sample(allImgs, args.numImages)

    reps = []
    for imgObj in imgObjs:
        rep = getRep(imgObj.path)

        if rep is not None:
            reps.append(rep)

    np.save(args.outputFile, np.row_stack(reps))
