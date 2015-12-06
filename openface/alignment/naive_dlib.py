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

import argparse
import cv2
import dlib
import numpy as np
import os
import random
import sys

from skimage import io

from .. import helper
from .. import data


class NaiveDlib:

    def __init__(self, faceMean, facePredictor):
        """Initialize the dlib-based alignment."""
        self.detector = dlib.get_frontal_face_detector()
        self.normMeanLandmarks = loadMeanPoints(faceMean)
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, img):
        return self.detector(img, 1)

    def getLargestFaceBoundingBox(self, img):
        faces = self.detector(img, 1)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())

    def align(self, img, bb):
        points = self.predictor(img, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    EYES_AND_NOSE = np.array([36, 45, 33])
    def alignImg(self, method, size, img, bb=None,
                 landmarks=None, landmarkIndices=EYES_AND_NOSE):
        if bb is None:
            try:
                bb = self.getLargestFaceBoundingBox(img)
            except Exception as e:
                print("Warning: {}".format(e))
                # In rare cases, exceptions are thrown.
                return
            if bb is None:
                # Most failed detection attempts return here.
                return

        if landmarks is None:
            landmarks = self.align(img, bb)

        npLandmarks = np.float32(landmarks)
        npNormMeanLandmarks = np.float32(self.normMeanLandmarks)

        if method == 'affine':
            H = cv2.getAffineTransform(npLandmarks[landmarkIndices],
                                       size*npNormMeanLandmarks[landmarkIndices])
            thumbnail = cv2.warpAffine(img, H, (size, size))
        else:
            raise Exception('Unrecognized method: {}'.format(method))

        return thumbnail

def transformPoints(points, bb, toImgCoords):
    if toImgCoords:
        def scale(p):
            (x, y) = p
            return (int((x * bb.width()) + bb.left()),
                    int((y * bb.height()) + bb.top()))
    else:
        def scale(p):
            (x, y) = p
            return (float(x - bb.left()) / bb.width(),
                    float(y - bb.top()) / bb.height())
    return list(map(scale, points))


def loadMeanPoints(modelFname):
    def parse(line):
        (x, y) = line.strip().split(",")
        return (float(x), float(y))
    with open(modelFname, 'r') as f:
        return [parse(line) for line in f]


def annotate(img, box, points=None, meanPoints=None):
    a = np.copy(img)
    bl = (box.left(), box.bottom())
    tr = (box.right(), box.top())
    cv2.rectangle(a, bl, tr, color=(153, 255, 204), thickness=3)
    for p in points:
        cv2.circle(a, center=p, radius=3, color=(102, 204, 255), thickness=-1)
    for p in meanPoints:
        cv2.circle(a, center=p, radius=3, color=(0, 0, 0), thickness=-1)
    return a
