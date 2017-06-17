#!/usr/bin/env python2

import argparse
import os

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('bmh')

parser = argparse.ArgumentParser()
parser.add_argument('workDir', type=str)
parser.add_argument('--names', type=str, nargs='+', required=True)
c_args = parser.parse_args()


def draw_2d(args):
    mpl.use('Agg')

    plt.style.use('bmh')

    out = "{}/tsne.pdf".format(args.workDir)
    if not os.path.exists(out):
        y = pd.read_csv("{}/labels.csv".format(args.workDir)).as_matrix()[:, 0]
        X = pd.read_csv("{}/reps.csv".format(args.workDir)).as_matrix()

        target_names = np.array(args.names)
        colors = cm.Dark2(np.linspace(0, 1, len(target_names)))

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        for c, i, target_name in zip(colors,
                                     list(range(1, len(target_names) + 1)),
                                     target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                        c=c, label=target_name)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
                   fontsize=8)

        plt.savefig(out)
        print("Saved to: {}".format(out))


def draw_3d(args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parser = argparse.ArgumentParser()
    parser.add_argument('workDir', type=str)
    parser.add_argument('--names', type=str, nargs='+', required=True)
    args = parser.parse_args()

    out = "{}/tsne_3d.pdf".format(args.workDir)
    if not os.path.exists(out):
        y = pd.read_csv("{}/labels.csv".format(args.workDir)).as_matrix()[:, 0]
        X = pd.read_csv("{}/reps.csv".format(args.workDir)).as_matrix()

        target_names = np.array(args.names)
        colors = cm.Dark2(np.linspace(0, 1, len(target_names)))

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=3, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        for c, i, target_name in zip(colors,
                                     list(range(1, len(target_names) + 1)),
                                     target_names):
            ax.scatter(X_r[y == i, 0], X_r[y == i, 1],
                       c=c, label=target_name)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
                   fontsize=8)

        plt.savefig(out)
        print("Saved to: {}".format(out))


draw_2d(c_args)
draw_3d(c_args)
