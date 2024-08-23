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
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)
import torch

import openface

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DLIB_MODEL_DIR = os.path.join(MODEL_DIR, 'dlib')
OPENFACE_MODEL_DIR = os.path.join(MODEL_DIR, 'openface')
IMG_DIM = 96


def get_rep(img_path):
    if args.verbose:
        print('Processing {}.'.format(img_path))
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise Exception('Unable to load image: {}'.format(img_path))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print('  + Original size: {}'.format(rgb_img.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgb_img)
    if bb is None:
        raise Exception('Unable to find a face: {}'.format(img_path))
    if args.verbose:
        print('  + Face detection took {} seconds.'.format(time.time() - start))

    start = time.time()
    aligned_face = align.align(IMG_DIM, rgb_img, bb,
                               landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise Exception('Unable to align image: {}'.format(img_path))
    if args.verbose:
        print('  + Face alignment took {} seconds.'.format(time.time() - start))

    start = time.time()

    aligned_face = (aligned_face / 255.).astype(np.float32)
    aligned_face = np.expand_dims(np.transpose(aligned_face, (2, 0, 1)), axis=0)  # BCHW order
    aligned_face = torch.from_numpy(aligned_face)
    if not args.cpu:
        aligned_face = aligned_face.to(torch.device('cuda'))

    rep = net.forward(aligned_face)
    rep = rep.cpu().detach().numpy().squeeze(0)

    if args.verbose:
        print('  + OpenFace forward pass took {} seconds.'.format(time.time() - start))
        print('Representation:')
        print(rep)
        print('-----\n')
    return rep


def compare(args):
    for (img1, img2) in itertools.combinations(args.imgs, 2):
        d = get_rep(img1) - get_rep(img2)
        print('Comparing {} with {}.'.format(img1, img2))
        print(
            '  + Squared l2 distance between representations: {:0.3f}'.format(np.dot(d, d)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', type=str, nargs='+', help='Input images.')
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(DLIB_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--dlibFaceDetectorType', type=str, choices=['HOG', 'CNN'],
                        help="Type of dlib's face detector to be used.", default='CNN')
    parser.add_argument('--dlibFaceDetector', type=str, help="Path to dlib's CNN face detector.",
                        default=os.path.join(DLIB_MODEL_DIR, 'mmod_human_face_detector.dat'))
    parser.add_argument('--networkModel', type=str, help='Path to pretrained OpenFace model.',
                        default=os.path.join(OPENFACE_MODEL_DIR, 'nn4.small2.v1.pt'))
    parser.add_argument('--cpu', action='store_true', help='Run OpenFace models on CPU only.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print('Argument parsing and loading libraries took {} seconds.'.format(
            time.time() - start))

    start = time.time()
    if args.dlibFaceDetectorType == 'CNN':
        align = openface.AlignDlib(args.dlibFacePredictor, args.dlibFaceDetector)
    else:
        align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.OpenFaceNet()
    if args.cpu:
        net.load_state_dict(torch.load(args.networkModel))
    else:
        net.load_state_dict(torch.load(args.networkModel, map_location='cuda'))
        net.to(torch.device('cuda'))
    net.eval()

    if args.verbose:
        print('Loading the dlib and OpenFace models took {} seconds.'.format(
            time.time() - start))

    compare(args)
