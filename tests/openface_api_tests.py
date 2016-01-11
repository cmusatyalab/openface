# OpenFace API tests.
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


import cv2
import os

import numpy as np
np.set_printoptions(precision=2)
from numpy.linalg import norm

import scipy
import scipy.spatial

import openface

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
modelDir = os.path.join(openfaceDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

exampleImages = os.path.join(openfaceDir, 'images', 'examples')
lfwSubset = os.path.join(openfaceDir, 'data', 'lfw-subset')

dlibFacePredictor = os.path.join(dlibModelDir,
                                 "shape_predictor_68_face_landmarks.dat")
nn4_v1_model = os.path.join(openfaceModelDir, 'nn4.v1.t7')
nn4_v2_model = os.path.join(openfaceModelDir, 'nn4.v2.t7')
imgDim = 96

align = openface.AlignDlib(dlibFacePredictor)
nn4_v1 = openface.TorchNeuralNet(nn4_v1_model, imgDim=imgDim)
nn4_v2 = openface.TorchNeuralNet(nn4_v2_model, imgDim=imgDim)


def test_v1_pipeline():
    imgPath = os.path.join(exampleImages, 'lennon-1.jpg')
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    assert np.isclose(norm(rgbImg), 11.1355)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    assert bb.left() == 341
    assert bb.right() == 1006
    assert bb.top() == 193
    assert bb.bottom() == 859

    # Should be INNER_EYES_AND_BOTTOM_LIP by default.
    alignedFace = align.align(imgDim, rgbImg, bb)
    assert np.isclose(norm(alignedFace), 8.30662)

    alignedFace_alt = align.align(imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    assert np.isclose(norm(alignedFace), norm(alignedFace_alt))


def test_v2_pipeline():
    imgPath = os.path.join(exampleImages, 'lennon-1.jpg')
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    assert np.isclose(norm(rgbImg), 11.1355)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    assert bb.left() == 341
    assert bb.right() == 1006
    assert bb.top() == 193
    assert bb.bottom() == 859

    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    assert np.isclose(norm(alignedFace), 7.61577)

    rep = nn4_v2.forward(alignedFace)
    cosDist = scipy.spatial.distance.cosine(rep, np.ones(128))
    assert np.isclose(cosDist, 0.981229293936)
