# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 10/12/2016 """
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import normalize

__author__ = 'cenk'


def create_table(data, rowLabels=None, colLabels=None, title=None, output=None):
    fig, axs = plt.subplots()
    # fig.set_size_inches(11.69, 8.27)
    # fig.set_size_inches(20, 18)
    axs.axis('tight')
    axs.axis('off')

    normed_matrix = normalize(data, axis=1, norm='max')
    normed_matrix[:, -1] = 1
    colors = plt.cm.gray(normed_matrix)

    axs.table(cellText=data, rowLabels=rowLabels, colLabels=colLabels, loc='center', cellColours=colors)
    plt.title(title)
    if output:
        plt.savefig(output + '/table.pdf', orientation='landscape', papertype='a2')
    else:
        plt.show()


def read_file(path, start=0):
    with open(path, 'rb') as f:
        scores = [float("{0:.4f}".format(float(score.replace('\n', '')))) for score in f.readlines()[start:]]
    return scores


def results2table(args):
    train_label = 'train'
    test_label = 'test'
    accuracy_label = 'accuracies.txt'
    path = args.workDir
    train_err_path = os.path.join(path, 'train.log')
    train_score_path = os.path.join(path, 'train_score.log')
    test_score_path = os.path.join(path, 'test_score.log')
    test_path = os.path.join(path, 'test.log')
    train_err = read_file(train_err_path, start=1)
    train_scores = read_file(train_score_path)
    test_scores = read_file(test_score_path)
    tests = read_file(test_path)

    results = []
    column_labels = []
    for i in range(1, len(train_scores) + 1):
        print i
        result = []
        rep_dir = os.path.join(path, 'rep-%s' % str(i))
        train_accuracy_dir = os.path.join(os.path.join(rep_dir, train_label), accuracy_label)
        test_accuracy_dir = os.path.join(os.path.join(rep_dir, test_label), accuracy_label)
        train_accuracies = read_file(train_accuracy_dir)
        test_accuracies = read_file(test_accuracy_dir)
        result.extend(train_accuracies)
        result.extend(test_accuracies)
        result.append(train_scores[i - 1])
        result.append(test_scores[i - 1])
        result.append(tests[i - 1])
        result.append(train_err[i - 1])
        results.append(result)
        if not column_labels:
            column_labels.extend(['Train %s' % j for j in range(1, len(train_accuracies) + 1)])
            column_labels.extend(['Test %s' % j for j in range(1, len(test_accuracies) + 1)])
            column_labels.append('Train Mean')
            column_labels.append('Test Mean 1')
            column_labels.append('Test Mean 2')
            column_labels.append('Train Error')
    row_labels = ['#Epoch %s' % j for j in range(1, len(train_scores) + 1)]
    results = np.array(results)

    create_table(results, rowLabels=row_labels, colLabels=column_labels, title=args.title, output=path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--workDir', type=str)
    parser.add_argument('--title', type=str)

    args = parser.parse_args()

    results2table(args)
