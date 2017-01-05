# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 19/11/2016 """
import operator

import numpy as np
import os
import pandas as pd
from sklearn import cross_validation
from sklearn import svm

from confusion_matrix import create_confusion_matrix

__author__ = 'cenk'


def classify(data_path, path=None):
    fname = "{}/labels.csv".format(data_path)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda x: x.split(".")[0], paths)
    paths = np.array(map(lambda path: os.path.splitext(path)[0], paths))

    fname = "{}/reps.csv".format(data_path)
    rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    folds = cross_validation.KFold(n=len(rawEmbeddings), n_folds=10, shuffle=True)
    scores, scores2, scores3 = [], [], []
    for idx, (train, test) in enumerate(folds):
        # clf = neighbors.KNeighborsClassifier(1)
        # clf.fit(rawEmbeddings_test[train], paths_test[train])
        # scores.append(clf.score(rawEmbeddings_test[test], paths_test[test]))
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(rawEmbeddings[train], paths[train])
        scores2.append(clf.score(rawEmbeddings[test], paths[test]))
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18,), random_state=1)
        # clf.fit(rawEmbeddings_test[train], paths_test[train])
        # scores3.append(clf.score(rawEmbeddings_test[test], paths_test[test]))
    accuracy_dir = os.path.abspath(os.path.join(data_path, 'accuracies.txt'))

    with open(accuracy_dir, "wb") as file:
        for i in scores2:
            file.writelines(str(i) + '\n')

    # print "Avg. score %s" % (reduce(operator.add, scores) / len(folds))
    print "Avg. score %s" % (reduce(operator.add, scores2) / len(folds))
    # print "Avg. score %s" % (reduce(operator.add, scores3) / len(folds))
    result_path = "{}/{}.log".format(os.path.abspath(os.path.join(os.path.join(data_path, os.pardir), os.pardir)), path)
    with open(result_path, "a") as file:
        file.write(
            str((reduce(operator.add, scores2) / len(folds))) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--testDir', type=str)

    args = parser.parse_args()

    classify(args.trainDir, path='train_score')
    create_confusion_matrix(args.trainDir, args.testDir,
                            os.path.abspath(os.path.join(args.trainDir, os.pardir)))

    classify(args.testDir, path='test_score')
