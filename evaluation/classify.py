# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 19/11/2016 """
import operator

import numpy as np
import os
import pandas as pd
from sklearn import svm, cross_validation, neighbors
from sklearn.neural_network import MLPClassifier

__author__ = 'cenk'


def classify(args):
    fname = "{}/labels.csv".format(args.workDir)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda x: x.split(".")[0], paths)
    paths = np.array(map(lambda path: os.path.splitext(path)[0], paths))

    fname = "{}/reps.csv".format(args.workDir)
    rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()

    folds = cross_validation.KFold(n=len(rawEmbeddings), n_folds=10, shuffle=True)
    scores = []
    scores2 = []
    scores3 = []
    for idx, (train, test) in enumerate(folds):
        clf = neighbors.KNeighborsClassifier(args.n)
        clf.fit(rawEmbeddings[train], paths[train])
        scores.append(clf.score(rawEmbeddings[test], paths[test]))
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(rawEmbeddings[train], paths[train])
        scores2.append(clf.score(rawEmbeddings[test], paths[test]))
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18,), random_state=1)
        clf.fit(rawEmbeddings[train], paths[train])
        scores3.append(clf.score(rawEmbeddings[test], paths[test]))

    print "Avg. score %s" % (reduce(operator.add, scores) / len(folds))
    print "Avg. score %s" % (reduce(operator.add, scores2) / len(folds))
    print "Avg. score %s" % (reduce(operator.add, scores3) / len(folds))
    result_path = "{}/test.log".format(os.path.abspath(os.path.join(args.workDir, os.pardir)))
    with open(result_path, "a") as file:
        file.write(str((reduce(operator.add, scores2) / len(folds))) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--workDir', type=str, help='an integer for the accumulator')
    parser.add_argument('-n', type=int, default=5)

    args = parser.parse_args()

    classify(args)
