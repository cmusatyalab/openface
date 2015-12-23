#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
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
import dlib
import itertools
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs
from openface.alignment import NaiveDlib

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(imgPath):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.alignImg("affine", args.imgDim, bgrImg, bb)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forwardImage(alignedFace)
    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(time.time() - start))
    return rep


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    param_grid = [
        {'C': [1, 10, 100, 1000],
            'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
            'gamma': [0.001, 0.0001],
            'kernel': ['rbf']}
    ]
    svm = GridSearchCV(
        SVC(probability=True),
        param_grid, verbose=4, cv=5, n_jobs=16
    ).fit(embeddings, labelsNum)
    print("Best estimator: {}".format(svm.best_estimator_))
    print("Best score on left out data: {:.2f}".format(svm.best_score_))

    with open("{}/classifier.pkl".format(args.workDir), 'w') as f:
        pickle.dump((le, svm), f)


def infer(args):
    with open(args.classifierModel, 'r') as f:
        (le, svm) = pickle.load(f)
    rep = getRep(args.img)
    start = time.time()
    predictions = svm.predict_proba(rep)[0]
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    if args.verbose:
        print("SVM prediction took {} seconds.".format(time.time() - start))
    print("Predict {} with {:.2f} confidence.".format(person, confidence))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('workDir', type=str,
                             help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser('infer',
                                        help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str,
                             help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('img', type=str,
                             help="Input image.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    align = NaiveDlib(args.dlibFacePredictor)
    net = openface.TorchWrap(
        args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
