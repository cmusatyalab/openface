#!/usr/bin/env python2
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

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import argparse
import cv2
import random
import shutil

from skimage import io

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def computeMeanMain(args):
    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)

    imgs = list(iterImgs(args.inputDir))
    if args.numImages > 0:
        imgs = random.sample(imgs, args.numImages)

    facePoints = []
    for img in imgs:
        rgb = img.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)
        alignedPoints = align.align(rgb, bb)
        if alignedPoints:
            facePoints.append(alignedPoints)

    facePointsNp = np.array(facePoints)
    mean = np.mean(facePointsNp, axis=0)
    std = np.std(facePointsNp, axis=0)

    write(mean, "{}/mean.csv".format(args.modelDir))
    write(std, "{}/std.csv".format(args.modelDir))

    # Only import in this mode.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(mean[:, 0], -mean[:, 1], color='k')
    ax.axis('equal')
    for i, p in enumerate(mean):
        ax.annotate(str(i), (p[0] + 0.005, -p[1] + 0.005), fontsize=8)
    plt.savefig("{}/mean.png".format(args.modelDir))


def alignMain(args):
    openface.helper.mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        outDir = os.path.join(args.outputDir, imgObject.cls)
        imgName = "{}/{}.png".format(outDir, imgObject.name)
        openface.helper.mkdirP(outDir)
        if not os.path.isfile(imgName):
            rgb = imgObject.getRGB(cache=False)
            out = align.alignImg(args.method, args.size, rgb,
                                 outputPrefix=outDir,
                                 outputDebug=args.outputDebugImages)
            if args.fallbackLfw and out is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,
                                                               imgObject.cls),
                                                  imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.outputDir,
                                                                          imgObject.cls),
                                                             imgObject.name))

            if out is not None:
                io.imsave(imgName, out)
    print('nFallbacks:', nFallbacks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "mean.csv"))
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--dlibRoot', type=str,
                        default=os.path.expanduser(
                            "~/src/dlib-18.16/python_examples"),
                        help="dlib directory with the dlib.so Python library.")

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    computeMeanParser = subparsers.add_parser(
        'computeMean', help='Compute the image mean of a directory of images.')
    computeMeanParser.add_argument('--numImages', type=int, help="The number of images. '0' for all images.",
                                   default=0)  # <= 0 ===> all imgs
    alignmentParser = subparsers.add_parser(
        'align', help='Align a directory of images.')
    alignmentParser.add_argument('method', type=str,
                                 choices=['tightcrop', 'affine',
                                          'perspective', 'homography'],
                                 help="Alignment method.")
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--outputDebugImages', action='store_true',
                                 help='Output annotated images for debugging and presenting.')
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=152)
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")

    args = parser.parse_args()

    sys.path.append(args.dlibRoot)
    import openface
    import openface.helper
    from openface.data import iterImgs
    from openface.alignment import NaiveDlib

    if args.mode == 'computeMean':
        computeMeanMain(args)
    else:
        alignMain(args)
