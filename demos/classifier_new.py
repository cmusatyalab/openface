#!/usr/bin/env python3
#
# Copyright 2015-2024 Carnegie Mellon University
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
import os
import pickle
import sys

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import torch

import openface

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import mixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DLIB_MODEL_DIR = os.path.join(MODEL_DIR, 'dlib')
OPENFACE_MODEL_DIR = os.path.join(MODEL_DIR, 'openface')
IMG_DIM = 96


def get_rep(img_path, multiple=False):
    start = time.time()
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise Exception('Unable to load image: {}'.format(img_path))

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print('  + Original size: {}'.format(rgb_img.shape))
    if args.verbose:
        print('Loading the image took {} seconds.'.format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgb_img)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgb_img)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception('Unable to find a face: {}'.format(img_path))
    if args.verbose:
        print('Face detection took {} seconds.'.format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        aligned_face = align.align(IMG_DIM, rgb_img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception('Unable to align image: {}'.format(img_path))
        if args.verbose:
            print('Alignment took {} seconds.'.format(time.time() - start))
            print('This bbox is centered at {}, {}'.format(bb.center().x, bb.center().y))

        start = time.time()

        aligned_face = (aligned_face / 255.).astype(np.float32)
        aligned_face = np.expand_dims(np.transpose(aligned_face, (2, 0, 1)), axis=0)  # BCHW order
        aligned_face = torch.from_numpy(aligned_face)
        if not args.cpu:
            aligned_face = aligned_face.to(torch.device('cuda'))
        rep = model(aligned_face)
        rep = rep.cpu().detach().numpy().squeeze(0)
        if args.verbose:
            print('Neural network forward pass took {} seconds.'.format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def train(args):
    print('Loading embeddings.')
    labels_file =  os.path.join(args.workDir, 'labels.csv')
    labels = pd.read_csv(labels_file, header=None).values[:, 1]
    labels = np.array(labels)
    reps_file = os.path.join(args.workDir, 'reps.csv')
    embeddings = pd.read_csv(reps_file, header=None).values
    le = LabelEncoder().fit(labels)
    labels_num = le.transform(labels)
    n_classes = len(le.classes_)
    print('Training for {} classes.'.format(n_classes))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif args.classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    elif args.classifier == 'GMM':  # Doesn't work best
        clf = mixture.GaussianMixture(n_components=n_classes)

    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

    # ref: https://jessesw.com/Deep-Learning/
    elif args.classifier == 'DBN':
        from nolearn.dbn import DBN
        clf = DBN([embeddings.shape[1], 500, labels_num[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                  learn_rates=0.3,
                  # Smaller steps mean a possibly more accurate result, but the
                  # training will take longer
                  learn_rate_decays=0.9,
                  # a factor the initial learning rate will be multiplied by
                  # after each iteration of the training
                  epochs=300,  # no of iternation
                  # dropouts = 0.25, # Express the percentage of nodes that
                  # will be randomly dropped as a decimal.
                  verbose=1)

    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labels_num)

    classifier_file = os.path.join(args.workDir, 'classifier.pkl')
    print('Saving classifier to "{}"'.format(classifier_file))
    with open(classifier_file, 'wb') as f:
        pickle.dump((le, clf), f)


def infer(args, multiple=False):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

    for img in args.imgs:
        print('\n=== {} ==='.format(img))
        reps = get_rep(img, multiple)
        if len(reps) > 1:
            print('List of faces in image from left to right')
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform([maxI])
            confidence = predictions[maxI]
            if args.verbose:
                print('Prediction took {} seconds.'.format(time.time() - start))
            if multiple:
                print('Predict {} @ x={} with {:.2f} confidence.'.format(str(person[0]), bbx,
                                                                         confidence))
            else:
                print('Predict {} with {:.2f} confidence.'.format(str(person[0]), confidence))
            if isinstance(clf, mixture.GaussianMixture):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print('  + Distance from the mean: {}'.format(dist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(DLIB_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--dlibFaceDetectorType', type=str, choices=['HOG', 'CNN'],
                        help="Type of dlib's face detector to be used.", default='CNN')
    parser.add_argument('--dlibFaceDetector', type=str, help="Path to dlib's CNN face detector.",
                        default=os.path.join(DLIB_MODEL_DIR, 'mmod_human_face_detector.dat'))
    parser.add_argument('--upsample', type=int, help="Number of times to upsample images before detection.", default=1)
    parser.add_argument('--networkModel', type=str, help='Path to pretrained OpenFace model.',
                        default=os.path.join(OPENFACE_MODEL_DIR, 'nn4.small2.v1.pt'))
    parser.add_argument('--cpu', action='store_true', help='Run OpenFace models on CPU only.')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    trainParser = subparsers.add_parser('train', help='Train a new classifier.')
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument('--classifier', type=str,
        choices=['LinearSvm',
                 'GridSearchSvm',
                 'GMM',
                 'RadialSvm',
                 'DecisionTree',
                 'GaussianNB',
                 'DBN'],
        help='The type of classifier to use.', default='LinearSvm')
    trainParser.add_argument('workDir', type=str,
                             help='The input work directory containing "reps.csv" and "labels.csv". Obtained from '
                                  'aligning a directory with "align-dlib" and getting the representations with '
                                  '"batch-represent".')

    inferParser = subparsers.add_parser('infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str,
                             help='The Python pickle representing the classifier. This is NOT the Torch network '
                                  'model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+', help='Input image.')
    inferParser.add_argument('--multi', help='Infer multiple faces in image', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print('Argument parsing and import libraries took {} seconds.'.format(
            time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith('.t7'):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    if args.dlibFaceDetectorType == 'CNN':
        align = openface.AlignDlib(args.dlibFacePredictor, args.dlibFaceDetector, upsample=args.upsample)
    else:
        align = openface.AlignDlib(args.dlibFacePredictor, upsample=args.upsample)
    model = openface.OpenFaceNet()
    if args.cpu:
        model.load_state_dict(torch.load(args.networkModel))
    else:
        model.load_state_dict(torch.load(args.networkModel, map_location='cuda'))
        model.to(torch.device('cuda'))
    model.eval()

    if args.verbose:
        print('Loading the dlib and OpenFace models took {} seconds.'.format(
            time.time() - start))
        start = time.time()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args, args.multi)
