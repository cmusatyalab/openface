# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 19/11/2016 """
import operator
import os
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn import cross_validation, neighbors
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from confusion_matrix import create_confusion_matrix
import gc

__author__ = 'cenk'


def classify(data_path, path=None, counter=None, alg='svm'):
    out = os.path.join(data_path, '%s_%s_%s' % (alg, path, 'confusion.png'))
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
    # print(rawEmbeddings.shape, paths.shape)
    folds = cross_validation.KFold(n=len(rawEmbeddings), random_state=1, n_folds=10, shuffle=True)
    scores = []
    fscores_weighted, fscores_macro, fscores_micro = [], [], []
    for idx, (train, test) in enumerate(folds):
        print idx, alg
        if alg == 'knn':
            clf = neighbors.KNeighborsClassifier(1)
        elif alg == 'svm':
            clf = svm.SVC(kernel='linear', C=1, max_iter=200000000)
            # clf = svm.LinearSVC()
            # clf = svm.SVC(kernel="poly", degree=5, C=1, verbose=10)
        elif alg == 'nn':
            # clf = MLPClassifier(random_state=2, max_iter=200000000)
            clf = MLPClassifier(random_state=2, max_iter=200000000, hidden_layer_sizes=(96, 64, 32))
        elif alg == 'nnd':
            # clf = MLPClassifier(random_state=2, max_iter=200000000)
            clf = MLPClassifier(random_state=2, max_iter=200000000)
        elif alg == 'poly':
            clf = svm.SVC(kernel="poly", max_iter=200000000)
        elif alg == 'rf':
            clf = RandomForestClassifier()
        clf.fit(rawEmbeddings[train], paths[train])
        gc.collect()
        score = clf.score(rawEmbeddings[test], paths[test])
        # print score, alg
        scores.append(score)
        prediction = clf.predict(rawEmbeddings[test])
        fscore_weighted = f1_score(paths[test], prediction, average="weighted")
        fscores_weighted.append(fscore_weighted)

        fscore_macro = f1_score(paths[test], prediction, average="macro")
        fscores_macro.append(fscore_macro)

        fscore_micro = f1_score(paths[test], prediction, average="micro")
        fscores_micro.append(fscore_micro)
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
    fscores_weighted_result_path = "{}/{}_{}_fscores_weighted.log".format(
        os.path.abspath(os.path.join(os.path.join(data_path, os.pardir), os.pardir)),
        path, alg)
    with open(fscores_weighted_result_path, "a") as file:
        file.write("%s,\t%s\t%s\n" % (str((reduce(operator.add, fscores_weighted) / len(folds))), str(counter), alg))

    fscores_macro_result_path = "{}/{}_{}_fscores_macro.log".format(
        os.path.abspath(os.path.join(os.path.join(data_path, os.pardir), os.pardir)),
        path, alg)
    with open(fscores_macro_result_path, "a") as file:
        file.write("%s,\t%s\t%s\n" % (str((reduce(operator.add, fscores_macro) / len(folds))), str(counter), alg))

    fscores_micro_result_path = "{}/{}_{}_fscores_micro.log".format(
        os.path.abspath(os.path.join(os.path.join(data_path, os.pardir), os.pardir)),
        path, alg)
    with open(fscores_micro_result_path, "a") as file:
        file.write("%s,\t%s\t%s\n" % (str((reduce(operator.add, fscores_micro) / len(folds))), str(counter), alg))


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

    start_classify.apply(args=(args.trainDir, args.testDir, args.pathName, args.train, args.counter, args.alg))
