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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

__author__ = 'cenk'


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, output=None,
                          path_name=None, alg=None):
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

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float("%.4f" % cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output:
        out = os.path.join(output, '%s_%s_%s' % (alg, path_name, 'confusion.png'))
        plt.savefig(out)
    else:
        plt.show()

def spark_svm(embeddings,labels):
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.regression import LabeledPoint

    # Load and parse the data
    def parsePoint(line):
        values = [float(x) for x in line.split(' ')]
        return LabeledPoint(values[0], values[1:])

    data = sc.textFile("data/mllib/sample_svm_data.txt")
    parsedData = data.map(parsePoint)

    # Build the model
    model = SVMWithSGD.train(parsedData, iterations=100)

    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))

def create_confusion_matrix(train_dir, test_dir, path_name=None, out_dir=None, alg='svm', counter=None):
    out = os.path.join(out_dir, '%s_%s_%s' % (alg, path_name, 'confusion.png'))
    if os.path.exists(out):
        return True
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
        clf = neighbors.KNeighborsClassifier(1)
    elif alg == 'nn':
        clf = MLPClassifier(random_state=2)
    elif alg == 'svm':
        clf = svm.SVC(kernel="linear")
    elif alg == 'poly':
        clf = svm.SVC(kernel="poly")
    elif alg == 'rf':
        clf = RandomForestClassifier()
    clf.fit(train_rawEmbeddings, train_paths)
    prediction = clf.predict(test_rawEmbeddings)
    score = clf.score(test_rawEmbeddings, test_paths)
    conf_mat = confusion_matrix(test_paths, prediction)

    labels = sorted(list(set(list(paths))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=out_dir, path_name=path_name, alg=alg)
    result_path = "{}/{}_{}.log".format(os.path.abspath(os.path.join(os.path.join(train_dir, os.pardir), os.pardir)),
                                        '%s_%s' % (path_name, 'test'), alg)
    with open(result_path, "a") as file:
        file.write("%s,\t%s\t%s\n" % (str(score), str(counter), alg))
    print(score)
