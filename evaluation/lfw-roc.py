#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn import cross_validation
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workDir', type=str, default='reps')
    args = parser.parse_args()

    print("Loading embeddings.")
    paths = pd.read_csv("{}/labels.csv".format(args.workDir)).as_matrix()[:,1]
    paths = map(os.path.basename, paths) # Get the filename.
    paths = map(lambda path: os.path.splitext(path)[0], paths) # Remove the extension.
    rawEmbeddings = pd.read_csv("{}/reps.csv".format(args.workDir)).as_matrix()
    embeddings = dict(zip(*[paths, rawEmbeddings]))

    pairs = loadPairs()
    analyze_accuracy(args.workDir, pairs, embeddings)
    plot_accuracy(args.workDir)

def loadPairs():
    print("  + Reading pairs.")
    pairs = []
    with open("/home/bamos/ofr/data/lfw/pairs.txt","r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            # if len(pair) == 3:
            #     pair[1] = int(pair[1])
            #     pair[2] = int(pair[2])
            # elif len(pair) == 4:
            #     pair[1] = int(pair[1])
            #     pair[3] = int(pair[3])
            pairs.append(pair)
    return pairs

def analyze_accuracy(workDir, pairs, embeddings):
    print("  + Computing accuracy.")
    from scipy import arange
    thresholds = arange(0,4,0.01)
    fname = workDir+"/unsupervised-l2-roc.csv"
    if os.path.isfile(fname):
        print("File exists, skipping: {}".format(fname))
        return
    with open(fname, "w") as f:
        f.write("threshold,tp,tn,fp,fn,tpr,fpr\n")
        print("tp,tn,fp,fn,tpr,fpr\n")
        for threshold in thresholds:
            tp=tn=fp=fn=0
            num_errors = 0
            for pair in pairs:
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
                # print(name1,name2)
                if name1 not in embeddings or name2 not in embeddings:
                    # Representation of one or both people is not available
                    # since sometimes the face cannot be aligned.
                    # Guess they are the same and note the error.
                    num_errors += 1
                    predict_same = True
                else:
                    vec1 = embeddings[name1]
                    vec2 = embeddings[name2]
                    diff = vec1-vec2
                    dist = np.dot(diff.T, diff)
                    predict_same = dist < threshold

                if predict_same and actual_same: tp += 1
                elif predict_same and not actual_same: fp += 1
                elif not predict_same and not actual_same: tn += 1
                elif not predict_same and actual_same: fn += 1

            # if tp+fn == 0 or fp+tn == 0:
            #     raise Exception("Unable to compute TPR or FPR.")
            if tp+fn == 0: tpr = 0
            else: tpr = float(tp)/float(tp+fn)
            if fp+tn == 0: fpr = 0
            else: fpr = float(fp)/float(fp+tn)
            print(threshold,tp,tn,fp,fn,tpr,fpr)
            f.write(",".join([str(x) for x in [threshold,tp,tn,fp,fn,tpr,fpr]]))
            f.write("\n")

    print("  + Errors: {}".format(num_errors))


def plot_accuracy(workDir):
    print("Plotting.")
    rocData = pd.read_csv("{}/unsupervised-l2-roc.csv".format(workDir))

    # plt.figure()
    fig, ax = plt.subplots(1,1)

    rocData.plot(x='fpr', y='tpr', ax=ax)

    openbrData = pd.read_csv("/tmp/openbr.DET.csv")
    openbrData['Y'] = 1-openbrData['Y']
    openbrData.plot(x='X', y='Y', legend=True, ax=ax)

    ax.legend(['FaceNet', 'OpenBR v1.0.0'], loc='lower right')

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
