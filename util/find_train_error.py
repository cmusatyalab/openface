# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """
import argparse

import os
import re
import pandas as pd

from util.create_tsne import create_tsne
from util.create_bar_chart import create_barchart

__author__ = 'cenk'

databases = {}
import numpy as np
import matplotlib.pyplot as plt

check_list = []


def find_train_errors(path):
    if path not in check_list:
        check_list.append(path)
        for _, folders, files in os.walk(path):
            found = False
            if files:
                for f in files:
                    if f == "train.log":
                        found = True
                        try:
                            absfile = os.path.join(path, f)
                            with open(absfile, mode="rb") as f_read:
                                lines = f_read.readlines()
                                lines = np.matrix(
                                    [[i * 10, float(line.replace("\t\n", ""))] for i, line in enumerate(lines[1:])])
                            splitted = path.split("/")
                            database = splitted[6]
                            loss = splitted[8]
                            network = splitted[9]
                            if database in databases:
                                databases[database].update({network: {}})
                            else:
                                databases[database] = {network: {}}

                            databases[database][network].update({'%s' % loss: lines})

                        except Exception as e:
                            print e.message

            if not found and folders:
                for folder in folders:
                    absfolder = os.path.join(path, folder)
                    find_train_errors(absfolder)


def draw_graphs(output):
    print "*0" * 100

    plt.figure(1)
    plt.subplot()
    for database, values in databases.iteritems():
        print database, values
        for network, value in values.iteritems():
            for loss, val in value.iteritems():
                plt.plot(val[:, 0], val[:, 1], label=loss)
            plt.xlabel("Epoch Count")
            plt.ylabel("Error")
            plt.title("%s-%s Training" % (database.upper(), network.upper()))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=3, fancybox=True)
            plt.savefig("%s/error_%s_%s.jpg" % (output, database, network), dpi=1200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    parser.add_argument('--output', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    find_train_errors(args.path)
    draw_graphs(args.output)
