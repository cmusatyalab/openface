#!/usr/bin/env python3
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

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import os
import sys

import argparse

from scipy import arange

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workDir', type=str, default='reps')
    args = parser.parse_args()

    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    paths = pd.read_csv(fname, header=None).as_matrix()[:,1]
    paths = map(os.path.basename, paths) # Get the filename.
    paths = map(lambda path: os.path.splitext(path)[0], paths) # Remove the extension.
    fname = "{}/reps.csv".format(args.workDir)
    rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()
    embeddings = dict(zip(*[paths, rawEmbeddings]))

    pairs = loadPairs()
    classifyExp(args.workDir, pairs, embeddings)
    plotClassifyExp(args.workDir)

def loadPairs():
    print("  + Reading pairs.")
    pairs = []
    with open("/home/bamos/ofr/data/lfw/pairs.txt","r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

def getEmbeddings(pair, embeddings):
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0],pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0],pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0],pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2],pair[3].zfill(4))
        actual_same = False
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))

    (x1, x2) = (embeddings[name1], embeddings[name2])
    return (x1, x2, actual_same)

def writeROC(fname, thresholds, embeddings, pairsTest):
    with open(fname,  "w") as f:
        f.write("threshold,tp,tn,fp,fn,tpr,fpr\n")
        tp=tn=fp=fn=0
        for threshold in thresholds:
            tp=tn=fp=fn=0
            for pair in pairsTest:
                (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
                diff = x1-x2
                dist = np.dot(diff.T, diff)
                predict_same = dist < threshold

                if predict_same and actual_same: tp += 1
                elif predict_same and not actual_same: fp += 1
                elif not predict_same and not actual_same: tn += 1
                elif not predict_same and actual_same: fn += 1

            if tp+fn == 0: tpr = 0
            else: tpr = float(tp)/float(tp+fn)
            if fp+tn == 0: fpr = 0
            else: fpr = float(fp)/float(fp+tn)
            f.write(",".join([str(x) for x in [threshold,tp,tn,fp,fn,tpr,fpr]]))
            f.write("\n")
            if tpr == 1.0 and fpr == 1.0:
                # No further improvements.
                f.write(",".join([str(x) for x in [4.0,tp,tn,fp,fn,tpr,fpr]]))
                return

def evalThresholdAccuracy(embeddings, pairs, threshold):
    y_true = []; y_predict = []
    for pair in pairs:
        (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
        diff = x1-x2
        dist = np.dot(diff.T, diff)
        predict_same = dist < threshold
        y_predict.append(predict_same)
        y_true.append(actual_same)

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy

def findBestThreshold(thresholds, embeddings, pairsTrain):
    bestThresh = bestThreshAcc = 0
    for threshold in thresholds:
        accuracy = evalThresholdAccuracy(embeddings, pairsTrain, threshold)
        if accuracy > bestThreshAcc:
            bestThreshAcc = accuracy
            bestThresh = threshold
        else:
            # No further improvements.
            return bestThresh
    return bestThresh

def classifyExp(workDir, pairs, embeddings):
    print("  + Computing accuracy.")
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = arange(0,4,0.01)

    if os.path.exists("{}/accuracies.txt".format(workDir)):
        print("{}/accuracies.txt already exists. Skipping processing.".format(workDir))
    else:
        accuracies = []
        with open("{}/accuracies.txt".format(workDir), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds):
                fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                writeROC(fname, thresholds, embeddings, pairs[test])

                bestThresh = findBestThreshold(thresholds, embeddings, pairs[train])
                accuracy = evalThresholdAccuracy(embeddings, pairs[test], bestThresh)
                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(idx, bestThresh, accuracy))
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(np.mean(accuracies),
                                                          np.std(accuracies)))


def plotClassifyExp(workDir):
    print("Plotting.")

    fig, ax = plt.subplots(1,1)

    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0,1,5000)
        fnFoldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)

    openbrData = pd.read_csv("comparisons/openbr.v1.0.0.DET.csv")
    openbrData['Y'] = 1-openbrData['Y']
    # brPlot = openbrData.plot(x='X', y='Y', legend=True, ax=ax)
    brPlot, = plt.plot(openbrData['X'], openbrData['Y'])

    fprs = [0]; tprs = [0]
    for fpr in np.linspace(0,1,5000):
        tpr = 0.0
        for f in fs:
            tpr += f(fpr)
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    fnMeanPlot, = plt.plot(fprs, tprs)

    humanData = pd.read_table("comparisons/kumar_human_crop.txt", header=None, sep=' ')
    humanPlot, = plt.plot(humanData[1], humanData[0])

    deepfaceData = pd.read_table("comparisons/deepface_ensemble.txt", header=None, sep=' ')
    dfPlot, = plt.plot(deepfaceData[1], deepfaceData[0], '--')

    baiduData = pd.read_table("comparisons/BaiduIDLFinal.TPFP", header=None, sep=' ')
    bPlot, = plt.plot(baiduData[1], baiduData[0])

    eigData = pd.read_table("comparisons/eigenfaces-original-roc.txt", header=None, sep=' ')
    eigPlot, = plt.plot(eigData[1], eigData[0])

    ax.legend([humanPlot, bPlot, dfPlot, brPlot, eigPlot, fnMeanPlot, fnFoldPlot],
              ['Human, Cropped', 'Baidu', 'DeepFace Ensemble', 'OpenBR v1.0.0',
               'Eigenfaces (no outside data)',
               'CMU FaceNet nn4.v1 mean', 'CMU FaceNet nn4.v1 folds'],
              loc='lower right')

    plt.plot([0,1], color='k', linestyle='dashed')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0,xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    fig.savefig(os.path.join(workDir, "roc.pdf"))

if __name__=='__main__':
    main()
