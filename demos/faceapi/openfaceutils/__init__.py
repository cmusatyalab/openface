# -*- coding: UTF-8 -*-

"""
@file __init__.py
@brief
    Defines for openface utils.

Created on: 2016/1/14
"""

import os

import openface
import argparse

import faceapi

"""
8888888b.            .d888 d8b
888  "Y88b          d88P"  Y8P
888    888          888
888    888  .d88b.  888888 888 88888b.   .d88b.  .d8888b
888    888 d8P  Y8b 888    888 888 "88b d8P  Y8b 88K
888    888 88888888 888    888 888  888 88888888 "Y8888b.
888  .d88P Y8b.     888    888 888  888 Y8b.          X88
8888888P"   "Y8888  888    888 888  888  "Y8888   88888P'
"""

modelDir = os.path.join(faceapi.BASE_DIR, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--dlibFacePredictor',
                    type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(
                                    dlibModelDir,
                                    "shape_predictor_68_face_landmarks.dat"))

parser.add_argument(
                    '--networkModel', type=str,
                    help="Path to Torch network model.",
                    default=os.path.join(
                                openfaceModelDir, 'nn4.small2.v1.t7'))

parser.add_argument(
                    '--imgDim', type=int,
                    help="Default image dimension.", default=96)

parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument(
                    '--unknown', type=bool, default=False,
                    help='Try to predict unknown people')

parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()
align = openface.AlignDlib(args.dlibFacePredictor)

neural_net = openface.TorchNeuralNet(
                                args.networkModel,
                                imgDim=args.imgDim,
                                cuda=args.cuda)
