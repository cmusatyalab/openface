#!/usr/bin/env python2
#
# This files can be used to benchmark different classifiers
# on lfw dataset with known and unknown dataset.
# More info at: https://github.com/cmusatyalab/openface/issues/144
# Brandon Amos & Vijayenthiran Subramaniam
# 2016/06/28
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
import os
import pickle

import shutil  # For copy images
import errno
import sys
import operator

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from nolearn.dbn import DBN

import multiprocessing

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

sys.path.append('./util/')
align_dlib = __import__('align-dlib')


# The list of available classifiers. The list is used in train() and
# inferFromTest() functions.
clfChoices = [
    'LinearSvm',
    'GMM',
    'RadialSvm',
    'DecisionTree',
    'GaussianNB',
    'DBN']


def train(args):
    start = time.time()
    for clfChoice in clfChoices:
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
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        if clfChoice == 'LinearSvm':
            clf = SVC(C=1, kernel='linear', probability=True)
        elif clfChoice == 'GMM':  # Doesn't work best
            clf = GMM(n_components=nClasses)

        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
        elif clfChoice == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif clfChoice == 'DecisionTree':  # Doesn't work best
            clf = DecisionTreeClassifier(max_depth=20)
        elif clfChoice == 'GaussianNB':
            clf = GaussianNB()

        # ref: https://jessesw.com/Deep-Learning/
        elif clfChoice == 'DBN':
            if args.verbose:
                verbose = 1
            else:
                verbose = 0
            clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=verbose)

        if args.ldaDim > 0:
            clf_final = clf
            clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                            ('clf', clf_final)])

        clf.fit(embeddings, labelsNum)

        fName = os.path.join(args.workDir, clfChoice + ".pkl")
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)
    if args.verbose:
        print(
            "Training and saving the classifiers took {} seconds.".format(
                time.time() - start))


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
    if (bb is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(
        args.imgDim,
        rgbImg,
        bb,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print(
            "Neural network forward pass took {} seconds.".format(
                time.time() - start))
    return rep


def inferFromTest(args):
    for clfChoice in clfChoices:
        print "==============="
        print "Using the classifier: " + clfChoice
        with open(os.path.join(args.featureFolder[0], clfChoice + ".pkl"), 'r') as f_clf:
            (le, clf) = pickle.load(f_clf)

        correctPrediction = 0
        inCorrectPrediction = 0
        sumConfidence = 0.0

        testSet = [
            os.path.join(
                args.testFolder[0], f) for f in os.listdir(
                args.testFolder[0]) if not f.endswith('.DS_Store')]

        for personSet in testSet:
            personImages = [os.path.join(personSet, f) for f in os.listdir(
                personSet) if not f.endswith('.DS_Store')]
            for img in personImages:
                if args.verbose:
                    print("\n=== {} ===".format(img.split('/')[-1:][0]))
                try:
                    rep = getRep(img).reshape(1, -1)
                except Exception as e:
                    print e
                    continue
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if args.verbose:
                    print(
                        "Prediction took {} seconds.".format(
                            time.time() - start))
                if args.verbose:
                    print(
                        "Predict {} with {:.2f} confidence.".format(
                            person, confidence))

                sumConfidence += confidence

                if confidence <= args.threshold and args.unknown:
                    person = "_unknown"

                if (img.split('/')[-1:][0].split('.')[0][:-5] == person and not args.unknown) or (person == "_unknown" and args.unknown):
                    correctPrediction += 1
                else:
                    inCorrectPrediction += 1

                if isinstance(clf, GMM) and args.verbose:
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))

        print "Results for the classifier: " + clfChoice
        print "Correct Prediction :" + str(correctPrediction)
        print "In-correct Prediction: " + str(inCorrectPrediction)
        print "Accuracy :" + str(float(correctPrediction) / (correctPrediction + inCorrectPrediction))
        print "Avg Confidence: " + str(float(sumConfidence) / (correctPrediction + inCorrectPrediction))


def preprocess(args):
    start = time.time()
    lfwPath = args.lfwDir
    destPath = args.featuresDir

    fullFaceDirectory = [os.path.join(lfwPath, f) for f in os.listdir(
        lfwPath) if not f.endswith('.DS_Store')]  # .DS_Store for the OS X

    noOfImages = []
    folderName = []

    for folder in fullFaceDirectory:
        try:
            noOfImages.append(len(os.listdir(folder)))
            folderName.append(folder.split('/')[-1:][0])
            # print folder.split('/')[-1:][0] +": " +
            # str(len(os.listdir(folder)))
        except:
            pass

    # Sorting
    noOfImages_sorted, folderName_sorted = zip(
        *sorted(zip(noOfImages, folderName), key=operator.itemgetter(0), reverse=True))

    with open(os.path.join(destPath, "List_of_folders_and_number_of_images.txt"), "w") as text_file:
        for f, n in zip(folderName_sorted, noOfImages_sorted):
            text_file.write("{} : {} \n".format(f, n))
    if args.verbose:
        print "Sorting lfw dataset took {} seconds.".format(time.time() - start)
        start = time.time()

    # Copy known train dataset
    for i in range(int(args.rangeOfPeople.split(':')[0]), int(
            args.rangeOfPeople.split(':')[1])):
        src = os.path.join(lfwPath, folderName_sorted[i])
        try:
            destFolder = os.path.join(
                destPath, 'train_known_raw', folderName_sorted[i])
            shutil.copytree(src, destFolder)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, destFolder)
            else:
                if args.verbose:
                    print('Directory not copied. Error: %s' % e)

    if args.verbose:
        print "Copying train dataset from lfw took {} seconds.".format(time.time() - start)
        start = time.time()

    # Take 10% images from train dataset as test dataset for known
    train_known_raw = [
        os.path.join(
            os.path.join(
                destPath,
                'train_known_raw'),
            f) for f in os.listdir(
            os.path.join(
                destPath,
                'train_known_raw')) if not f.endswith('.DS_Store')]  # .DS_Store for the OS X
    for folder in train_known_raw:
        images = [os.path.join(folder, f) for f in os.listdir(
            folder) if not f.endswith('.DS_Store')]
        if not os.path.exists(os.path.join(
                destPath, 'test_known_raw', folder.split('/')[-1:][0])):
            os.makedirs(os.path.join(destPath, 'test_known_raw',
                                     folder.split('/')[-1:][0]))
            # print "Created {}".format(os.path.join(destPath,
            # 'test_known_raw', folder.split('/')[-1:][0]))
        for i in range(int(0.9 * len(images)), len(images)):
            destFile = os.path.join(destPath, 'test_known_raw', folder.split(
                '/')[-1:][0], images[i].split('/')[-1:][0])
            try:
                shutil.move(images[i], destFile)
            except:
                pass
    if args.verbose:
        print "Spliting lfw dataset took {} seconds.".format(time.time() - start)
        start = time.time()

    # Copy unknown test dataset
    for i in range(int(args.rangeOfPeople.split(':')
                       [1]), len(folderName_sorted)):
        src = os.path.join(lfwPath, folderName_sorted[i])
        try:
            destFolder = os.path.join(
                destPath, 'test_unknown_raw', folderName_sorted[i])
            shutil.copytree(src, destFolder)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, destFolder)
            else:
                if args.verbose:
                    print('Directory not copied. Error: %s' % e)

    if args.verbose:
        print "Copying test dataset from lfw took {} seconds.".format(time.time() - start)
        start = time.time()

    class Args():
        """
            This class is created to pass arguments to ./util/align-dlib.py
        """

        def __init__(self, inputDir, outputDir, verbose):
            self.inputDir = inputDir
            self.dlibFacePredictor = os.path.join(
                dlibModelDir, "shape_predictor_68_face_landmarks.dat")
            self.mode = 'align'
            self.landmarks = 'outerEyesAndNose'
            self.size = 96
            self.outputDir = outputDir
            self.skipMulti = True
            self.verbose = verbose
            self.fallbackLfw = False

    argsForAlign = Args(
        os.path.join(
            destPath,
            'train_known_raw'),
        os.path.join(
            destPath,
            'train_known_aligned'),
        args.verbose)

    jobs = []
    for i in range(8):
        p = multiprocessing.Process(
            target=align_dlib.alignMain, args=(
                argsForAlign,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    if args.verbose:
        print "Aligning the raw train data took {} seconds.".format(time.time() - start)
        start = time.time()

    os.system(
        './batch-represent/main.lua -outDir ' +
        os.path.join(
            destPath,
            'train_known_features') +
        ' -data ' +
        os.path.join(
            destPath,
            'train_known_aligned'))

    if args.verbose:
        print "Extracting features from aligned train data took {} seconds.".format(time.time() - start)
        start = time.time()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree'],
        help='The type of classifier to use.',
        default='LinearSvm')
    trainParser.add_argument(
        'workDir',
        type=str,
        help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")

    inferFromTestParser = subparsers.add_parser(
        'inferFromTest',
        help='Predict who an image contains from a trained classifier.')
#    inferFromTestParser.add_argument('--classifierModel', type=str,
# help='The Python pickle representing the classifier. This is NOT the
# Torch network model, which can be set with --networkModel.')
    inferFromTestParser.add_argument(
        'featureFolder',
        type=str,
        nargs='+',
        help="Input the fratures folder which has the classifiers.")
    inferFromTestParser.add_argument(
        'testFolder',
        type=str,
        nargs='+',
        help="Input the test folder. It can be either known test dataset or unknown test dataset.")

    inferFromTestParser.add_argument(
        '--threshold',
        type=float,
        nargs='+',
        help="Threshold of the confidence to classify a prediction as unknown person. <threshold will be predicted as unknown person.",
        default=0.0)

    inferFromTestParser.add_argument(
        '--unknown',
        action='store_true',
        help="Use this flag if you are testing on unknown dataset. Make sure you set thresold value")

    preprocessParser = subparsers.add_parser(
        'preprocess',
        help='Before Benchmarking preprocess divides the dataset into train and test pairs. Also it will align the train dataset and extract the features from it.')

    preprocessParser.add_argument('--lfwDir', type=str,
                                  help="Enter the lfw face directory")

    preprocessParser.add_argument(
        '--rangeOfPeople',
        type=str,
        help="Range of the people you would like to take as known person group. Not that the input is a list starts with 0 and the people are sorted in decending order of number of images. Eg: 0:10 ")

    preprocessParser.add_argument(
        '--featuresDir',
        type=str,
        help="Enter the directory location where the aligned images, features, and classifer model will be saved.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        # infer(args)
        raise Exception("Use ./demo/classifier.py")
    elif args.mode == 'inferFromTest':
        inferFromTest(args)
    elif args.mode == 'preprocess':
        preprocess(args)
