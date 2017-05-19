# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 19/11/2016 """
import operator
import os

import numpy as np
import pandas as pd
from sklearn import cross_validation, neighbors
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from confusion_matrix import create_confusion_matrix

__author__ = 'cenk'


def classify(data_path, path=None, counter=None, alg='svm'):
    out = os.path.join(output, '%s_%s_%s' % (alg, path_name, 'confusion.png'))
    if os.path.exists(out):
        return True
    fname = "{}/labels.csv".format(data_path)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda x: x.split(".")[0], paths)
    paths = np.array(map(lambda path: os.path.splitext(path)[0], paths))

    fname = "{}/reps.csv".format(data_path)
    rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()

    folds = cross_validation.KFold(n=len(rawEmbeddings), random_state=1, n_folds=10, shuffle=True)
    scores = []
    for idx, (train, test) in enumerate(folds):
        if alg == 'knn':
            clf = neighbors.KNeighborsClassifier(1)
        elif alg == 'svm':
            n_estimators = 50
            # clf = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', C=1,cache_size=7000), max_samples=1.0 / n_estimators,
            #                                            n_estimators=n_estimators))
            # clf = svm.LinearSVC()
            clf = svm.SVC(kernel="linear", C=1)
        elif alg == 'nn':
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18,), random_state=1)
        elif alg == 'rf':
            clf = RandomForestClassifier(min_samples_leaf=20)
        clf.fit(rawEmbeddings[train], paths[train])
        scores.append(clf.score(rawEmbeddings[test], paths[test]))
    accuracy_dir = os.path.abspath(os.path.join(data_path, 'accuracies_%s.txt' % alg))

    with open(accuracy_dir, "wb") as file:
        for i in scores:
            file.writelines("%s,%s\n" % (str(i), str(counter)))
    # print "KNN Avg. score %s" % (reduce(operator.add, scores) / len(folds))
    # print "MLP Avg. score %s" % (reduce(operator.add, scores3) / len(folds))
    print "Avg. score %s" % (reduce(operator.add, scores) / len(folds)), data_path
    result_path = "{}/{}_{}.log".format(os.path.abspath(os.path.join(os.path.join(data_path, os.pardir), os.pardir)),
                                        path, alg)
    with open(result_path, "a") as file:
        file.write("%s,\t%s\t%s\n" % (str((reduce(operator.add, scores) / len(folds))), str(counter), alg))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--testDir', type=str)
    parser.add_argument('--pathName', type=str)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--counter', type=int, default=0)
    parser.add_argument('--alg', type=str, default='svm')
    args = parser.parse_args()
    from tasks import start_classify

    start_classify.apply_async(args=(args.trainDir, args.testDir, args.pathName, args.train, args.counter, args.alg))
