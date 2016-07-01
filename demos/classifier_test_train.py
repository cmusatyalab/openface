#!/usr/bin/env python
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
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

fileDir = os.path.dirname(os.path.realpath(__file__))
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
        print "Unable to find a face: {}".format(imgPath)
        return None
        #raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align_v1(args.imgDim, rgbImg, bb,
                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("******** Unable to align image: {} ***********".format(imgPath))
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))
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
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C = 1, kernel = 'linear', probability = True)
    elif args.classifier == 'GMM': #Doesn't work best
        clf = GMM(n_components=nClasses)


    #ref: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif args.classifier == 'RadialSvm': #Radial Basis Function kernel
        clf = SVC(C = 1000, kernel = 'rbf', probability = True, gamma = 0.05) #works better with C = 1 and gamma = 2
    elif args.classifier == 'DecisionTree': #Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)


    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])
    
    print "Embeddings: "
    print embeddings.shape
    print "\nlabelsNum: "
    print labelsNum[-1:][0] + 1

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def infer(args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    for img in args.imgs:
        print("\n=== {} ===".format(img))
        try:
            rep = getRep(img).reshape(1, -1)
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
        except:
            pass

#Added - 0628

def inferFromTest(args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    
    correctPrediction = 0
    inCorrectPrediction = 0
    highestConfidence = 0.0
    sumConfidence = 0.0

    testSet = [os.path.join(args.testFolder[0], f) for f in os.listdir(args.testFolder[0]) if not f.endswith('.DS_Store')]
    
    for personSet in testSet:
        personImages = [os.path.join(personSet, f) for f in os.listdir(personSet) if not f.endswith('.DS_Store')]
        for img in personImages:
            print("\n=== {} ===".format(img.split('/')[-1:][0]))
            try:
                rep = getRep(img).reshape(1, -1)
            except:
                continue
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
            
            if confidence > highestConfidence:
                highestConfidence = confidence
            
            sumConfidence += confidence
            
            if confidence <= args.threshold and args.unknown == True:
                person = "_unknown"
            
            if (img.split('/')[-1:][0].split('.')[0][:-5] == person and args.unknown == False) or (person == "_unknown" and args.unknown== True):
                correctPrediction += 1
            else:
                inCorrectPrediction += 1
            
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
        print "correctPrediction :" + str(correctPrediction)
        print "inCorrectPrediction: " + str(inCorrectPrediction)

    print "Accuracy :" + str(float(correctPrediction)/(correctPrediction+inCorrectPrediction))
    print "Highest Confidence: " + str(highestConfidence)
    print "Avg Confidence: " + str(float(sumConfidence)/(correctPrediction+inCorrectPrediction))

#Added 0629
def benchmark(args):
    import shutil #For copy images
    import errno
    import sys
    import operator
    
    lfwPath = args.lfwDir
    destPath = args.featuresDir

    fullFaceDirectory = [os.path.join(lfwPath, f) for f in os.listdir(lfwPath) if not f.endswith('.DS_Store')] #.DS_Store for the OS X

    noOfImages = []
    folderName = []

    for folder in image_paths:
        try:
            folderName.append(folder.split('/')[-1:][0])
            noOfImages.append(len(os.listdir(folder)))
            #print folder.split('/')[-1:][0] +": " + str(len(os.listdir(folder)))
        except:
            pass

    noOfImages_sorted, folderName_sorted = zip(*sorted(zip(noOfImages, folderName), key=operator.itemgetter(0), reverse=True))

    with open(os.path.join(destPath, "List_of_folders_and_number_of_images.txt"), "w") as text_file:
        for f,n in zip(folderName_sorted,noOfImages_sorted):
            text_file.write("{} : {} \n".format(f,n))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument('--classifier', type=str,
                             choices=['LinearSvm', 'GMM', 'RadialSvm', 'DecisionTree'],
                             help='The type of classifier to use.',
                             default='LinearSvm')
    trainParser.add_argument('workDir', type=str,
                             help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser('infer',
                                        help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str,
                             help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")
    
    #Added - 0628
    inferFromTestParser = subparsers.add_parser('inferFromTest',
                                     help='Predict who an image contains from a trained classifier.')
    inferFromTestParser.add_argument('classifierModel', type=str,
                          help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferFromTestParser.add_argument('testFolder', type=str, nargs='+',
                          help="Input the test folder.")
                          
    inferFromTestParser.add_argument('--threshold', type=float, nargs='+',
                                   help="Input the test folder.",
                                     default=0.0)
    
    inferFromTestParser.add_argument('--unknown', action='store_true')

    #Added - 0629
    
    benchmarkParser = subparsers.add_parser('benchmark',
                                            help='Benchmark a classifier based on the lfw dataset with known and unknown people.')
    
    benchmarkParser.add_argument('--lfwDir', type=str,
                                 help='Enter the lfw face directory')
    
    
    benchmarkParser.add_argument('--rangeOfPeople', type=str,
                                     help='Range of the people you would like to take as known person group. Not that the input is a list starts with 0 and the people are sorted in decending order of number of images')
    
    benchmarkParser.add_argument('--classifier', type=str,
                                 choices=['LinearSvm', 'GMM', 'RadialSvm', 'DecisionTree'],
                                 help='The type of classifier to use.',
                                 default='LinearSvm')
                                 
    benchmarkParser.add_argument('--featuresDir', type=str,
                                 help='Enter the directory location where the aligned images, features, and classifer model will be saved')
                                 

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    if (args.mode == 'infer' or args.mode == 'inferFromTest') and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
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
        infer(args)
    elif args.mode == 'inferFromTest':
        inferFromTest(args)
    elif args.mode == 'benchmark':
        benchmark(Args)
