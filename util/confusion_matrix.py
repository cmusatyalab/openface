# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 01/12/2016 """
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import svm, neighbors
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

__author__ = 'cenk'


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, output=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15, 15), dpi=80)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float("%.4f" % cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output:
        out = os.path.join(output, 'confusion.png')
        plt.savefig(out)
        print("Plot saved to %s" % out)
    else:
        plt.show()


def create_confusion_matrix(train_dir, test_dir, out_dir=None, alg='svm'):
    fname = "{}/labels.csv".format(train_dir)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda x: x.split(".")[0], paths)
    train_paths = np.array(map(lambda path: os.path.splitext(path)[0], paths))

    fname = "{}/reps.csv".format(train_dir)
    train_rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()

    fname = "{}/labels.csv".format(test_dir)
    paths = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    paths = map(os.path.basename, paths)  # Get the filename.
    # Remove the extension.
    paths = map(lambda x: x.split(".")[0], paths)
    test_paths = np.array(map(lambda path: os.path.splitext(path)[0], paths))

    fname = "{}/reps.csv".format(test_dir)
    test_rawEmbeddings = pd.read_csv(fname, header=None).as_matrix()

    if alg == 'knn':
        print("Using KNN")
        clf = neighbors.KNeighborsClassifier(1)
    elif alg == 'nn':
        print("Using NN")
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18,), random_state=1)
    else:
        print("Using SVM")
        clf = svm.SVC(kernel='linear', C=1)
    clf.fit(train_rawEmbeddings, train_paths)
    prediction = clf.predict(test_rawEmbeddings)
    print(clf.score(test_rawEmbeddings, test_paths))
    conf_mat = confusion_matrix(test_paths, prediction)

    labels = sorted(list(set(list(paths))))

    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--testDir', type=str)
    parser.add_argument('--outDir', type=str)
    parser.add_argument('--alg', type=str)

    args = parser.parse_args()

    create_confusion_matrix(args.trainDir, args.testDir, args.outDir, args.alg)
