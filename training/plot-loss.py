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

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd

import os

scriptDir = os.path.dirname(os.path.realpath(__file__))
plotDir = os.path.join(scriptDir, 'plots')
workDir = os.path.join(scriptDir, 'work')


def plot(workDirs):
    trainDfs = []
    # testDfs = []
    for d in workDirs:
        trainF = os.path.join(workDir, str(d), 'train.log')
        # testF = os.path.join(workDir, str(d), 'test.log')
        trainDfs.append(pd.read_csv(trainF, sep='\t'))
        # testDfs.append(pd.read_csv(testF, sep='\t'))
        # if len(trainDfs[-1]) != len(testDfs[-1]):
        #     print("Error: Train/test dataframe shapes "
        #           "for '{}' don't match: {}, {}".format(
        #               d, trainDfs[-1].shape, testDfs[-1].shape))
        #     sys.exit(-1)
    trainDf = pd.concat(trainDfs, ignore_index=True)
    # testDf = pd.concat(testDfs, ignore_index=True)

    # print("train, test:")
    # print("\n".join(["{:0.2e}, {:0.2e}".format(x, y) for (x, y) in
    #                  zip(trainDf['avg triplet loss (train set)'].values[-5:],
    #                      testDf['avg triplet loss (test set)'].values[-5:])]))

    fig, ax = plt.subplots(1, 1)
    trainDf.index += 1
    # testDf.index += 1
    trainDf['avg triplet loss (train set)'].plot(legend='True', ax=ax)
    # testDf['avg triplet loss (test set)'].plot(legend='True', ax=ax, alpha=0.6)
    plt.legend(['Train loss, semi-hard triplets'])  # 'Test loss, random triplets'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(ymin=0)
    plt.xlim(xmin=1)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    ax.set_yscale('log')
    fig.savefig(os.path.join(plotDir, "loss.pdf"))

if __name__ == '__main__':
    os.makedirs(plotDir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('workDirs', type=int, nargs='+')
    args = parser.parse_args()
    plot(args.workDirs)
