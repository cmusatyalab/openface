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
#
# This file implements a non-standard LFW classification experiment for
# the purposes of benchmarking the performance and accuracies of
# classification techniques.
# For the standard LFW experiment, see lfw.py.

import cv2
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import operator
import os
import pickle
import sys
import time

import argparse

import openface

sys.path.append("..")
from openface.helper import mkdirP

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

nPplVals = [10, 25, 50, 100]
nImgs = 20

cmap = plt.get_cmap("Set1")
colors = cmap(np.linspace(0, 0.5, 5))
alpha = 0.7


def main():
    parser = argparse.ArgumentParser()
    lfwDefault = os.path.expanduser(
        "~/openface/data/lfw/dlib.affine.sz:96.OuterEyesAndNose")
    parser.add_argument('--lfwAligned', type=str,
                        default=lfwDefault,
                        help='Location of aligned LFW images')
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--largeFont', action='store_true')
    parser.add_argument('workDir', type=str,
                        help='The work directory where intermediate files and results are kept.')
    args = parser.parse_args()
    # print(args)

    if args.largeFont:
        font = {'family': 'normal', 'size': 20}
        mpl.rc('font', **font)

    mkdirP(args.workDir)

    print("Getting lfwPpl")
    lfwPplCache = os.path.join(args.workDir, 'lfwPpl.pkl')
    lfwPpl = cacheToFile(lfwPplCache)(getLfwPplSorted)(args.lfwAligned)

    print("Eigenfaces Experiment")
    cls = cv2.createEigenFaceRecognizer()
    cache = os.path.join(args.workDir, 'eigenFacesExp.pkl')
    eigenFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    print("Fisherfaces Experiment")
    cls = cv2.createFisherFaceRecognizer()
    cache = os.path.join(args.workDir, 'fisherFacesExp.pkl')
    fishFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    print("LBPH Experiment")
    cls = cv2.createLBPHFaceRecognizer()
    cache = os.path.join(args.workDir, 'lbphExp.pkl')
    lbphFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    print("OpenFace CPU/SVM Experiment")
    net = openface.TorchNeuralNet(args.networkModel, 96, cuda=False)
    cls = SVC(kernel='linear', C=1)
    cache = os.path.join(args.workDir, 'openface.cpu.svm.pkl')
    openfaceCPUsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    print("OpenFace GPU/SVM Experiment")
    net = openface.TorchNeuralNet(args.networkModel, 96, cuda=True)
    cache = os.path.join(args.workDir, 'openface.gpu.svm.pkl')
    openfaceGPUsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    plotAccuracy(args.workDir, args.largeFont,
                 eigenFacesDf, fishFacesDf, lbphFacesDf,
                 openfaceCPUsvmDf, openfaceGPUsvmDf)
    plotTrainingTime(args.workDir, argrs.largeFont,
                     eigenFacesDf, fishFacesDf, lbphFacesDf,
                     openfaceCPUsvmDf, openfaceGPUsvmDf)
    plotPredictionTime(args.workDir, args.largeFont,
                       eigenFacesDf, fishFacesDf, lbphFacesDf,
                       openfaceCPUsvmDf, openfaceGPUsvmDf)

# http://stackoverflow.com/questions/16463582


def cacheToFile(file_name):
    def decorator(original_func):
        global cache
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except:
            cache = None

        def new_func(*param):
            global cache
            if cache is None:
                cache = original_func(*param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache
        return new_func

    return decorator


def getLfwPplSorted(lfwAligned):
    lfwPpl = {}
    for person in os.listdir(lfwAligned):
        fullPath = os.path.join(lfwAligned, person)
        if os.path.isdir(fullPath):
            nFiles = len([item for item in os.listdir(fullPath)
                          if os.path.isfile(os.path.join(fullPath, item))])
            lfwPpl[fullPath] = nFiles
    return sorted(lfwPpl.items(), key=operator.itemgetter(1), reverse=True)


def getData(lfwPpl, nPpl, nImgs, mode):
    X, y = [], []

    personNum = 0
    for (person, nTotalImgs) in lfwPpl[:nPpl]:
        imgs = sorted(os.listdir(person))
        for imgPath in imgs[:nImgs]:
            imgPath = os.path.join(person, imgPath)
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (96, 96))
            if mode == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                assert 0

            X.append(img)
            y.append(personNum)

        personNum += 1

    X = np.array(X)
    y = np.array(y)
    return (X, y)


def opencvExp(lfwAligned, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs', 'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))

    df_i = 0
    for nPpl in nPplVals:
        print(" + nPpl: {}".format(nPpl))
        (X, y) = getData(lfwAligned, nPpl, nImgs, mode='grayscale')
        nSampled = X.shape[0]
        ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

        allTrainTimeSec = []
        allPredictTimeSec = []
        accs = []

        for train, test in ss:
            start = time.time()
            cls.train(X[train], y[train])
            trainTimeSec = time.time() - start
            allTrainTimeSec.append(trainTimeSec)

            y_predict = []
            for img in X[test]:
                start = time.time()
                (label, score) = cls.predict(img)
                y_predict.append(label)
                predictTimeSec = time.time() - start
                allPredictTimeSec.append(predictTimeSec)
            y_predict = np.array(y_predict)

            acc = accuracy_score(y[test], y_predict)
            accs.append(acc)

        df.loc[df_i] = [nPpl, nImgs,
                        np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                        np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                        np.mean(accs), np.std(accs)]
        df_i += 1

    return df


def openfaceExp(lfwAligned, net, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs',
                               'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))

    repCache = {}

    df_i = 0
    for nPpl in nPplVals:
        print(" + nPpl: {}".format(nPpl))
        (X, y) = getData(lfwAligned, nPpl, nImgs, mode='rgb')
        nSampled = X.shape[0]
        ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

        allTrainTimeSec = []
        allPredictTimeSec = []
        accs = []

        for train, test in ss:
            X_train = []
            for img in X[train]:
                h = hash(str(img.data))
                if h in repCache:
                    rep = repCache[h]
                else:
                    rep = net.forward(img)
                    repCache[h] = rep
                X_train.append(rep)

            start = time.time()
            X_train = np.array(X_train)
            cls.fit(X_train, y[train])
            trainTimeSec = time.time() - start
            allTrainTimeSec.append(trainTimeSec)

            start = time.time()
            X_test = []
            for img in X[test]:
                X_test.append(net.forward(img))
            y_predict = cls.predict(X_test)
            predictTimeSec = time.time() - start
            allPredictTimeSec.append(predictTimeSec / len(test))
            y_predict = np.array(y_predict)

            acc = accuracy_score(y[test], y_predict)
            accs.append(acc)

        df.loc[df_i] = [nPpl, nImgs,
                        np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                        np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                        np.mean(accs), np.std(accs)]
        df_i += 1

    return df


def plotAccuracy(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                 openfaceCPUsvmDf, openfaceGPUsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.2

    if largeFont:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['accsMean'], barWidth,
            yerr=eigenFacesDf['accsStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['accsMean'], barWidth,
            yerr=fishFacesDf['accsStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['accsMean'], barWidth,
            yerr=lbphFacesDf['accsStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUsvmDf['accsStd'], label='OpenFace',
            color=colors[3], ecolor='0.3', alpha=alpha)

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.07, box.width, box.height * 0.83])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.85])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4,
                   fancybox=True, shadow=True)
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Number of People")

    ax.set_xticks(indices + 2 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)

    locs, labels = plt.xticks()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(workDir, 'accuracies.png'))


def plotTrainingTime(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                     openfaceCPUsvmDf, openfaceGPUsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.2

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['trainTimeSecMean'], barWidth,
            yerr=eigenFacesDf['trainTimeSecStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['trainTimeSecMean'], barWidth,
            yerr=fishFacesDf['trainTimeSecStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['trainTimeSecMean'], barWidth,
            yerr=lbphFacesDf['trainTimeSecStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['trainTimeSecStd'],
            label='OpenFace',
            color=colors[3], ecolor='0.3', alpha=alpha)

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.08, box.width, box.height * 0.83])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=4,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.85])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4,
                   fancybox=True, shadow=True)
    plt.ylabel("Training Time (s)")
    plt.xlabel("Number of People")

    ax.set_xticks(indices + 2 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)
    locs, labels = plt.xticks()
    # plt.setp(labels, rotation=45)
    # plt.ylim(0, 1)

    ax.set_yscale('log')
    plt.savefig(os.path.join(workDir, 'trainTimes.png'))


def plotPredictionTime(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                       openfaceCPUsvmDf, openfaceGPUsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.15

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['predictTimeSecMean'], barWidth,
            yerr=eigenFacesDf['predictTimeSecStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['predictTimeSecMean'], barWidth,
            yerr=fishFacesDf['predictTimeSecStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['predictTimeSecMean'], barWidth,
            yerr=lbphFacesDf['predictTimeSecStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['predictTimeSecStd'],
            label='OpenFace CPU',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfaceGPUsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceGPUsvmDf['predictTimeSecStd'],
            label='OpenFace GPU',
            color=colors[4], ecolor='0.3', alpha=alpha)

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.11, box.width, box.height * 0.7])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.77])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37), ncol=3,
                   fancybox=True, shadow=True)
    plt.ylabel("Prediction Time (s)")
    plt.xlabel("Number of People")
    ax.set_xticks(indices + 2.5 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)
    ax.xaxis.grid(False)
    locs, labels = plt.xticks()
    # plt.setp(labels, rotation=45)
    # plt.ylim(0, 1)

    ax.set_yscale('log')
    plt.savefig(os.path.join(workDir, 'predictTimes.png'))

if __name__ == '__main__':
    main()
