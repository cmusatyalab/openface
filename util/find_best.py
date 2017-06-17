# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/10/2016 """
import argparse

import os
import re
import pandas as pd

from util.create_tsne import create_tsne
from util.create_bar_chart import create_barchart

__author__ = 'cenk'


def find_best(path, output, train=False):
    reg = r"(\d+\.\d+),\s(\d+)\s(\w+)"
    for _, folders, files in os.walk(path):
        if folders:
            for folder in folders:
                absfolder = os.path.join(path, folder)
                find_best(absfolder, output, train=train)
        if files:
            for f in files:
                p = 'test_score'
                if train:
                    p = 'train'
                if f.endswith('.log') and p in f:#or ("test_score_svm" in f and "mnist" in path)):
                    print f
                    try:
                        absfile = os.path.join(path, f)
                        arr = []
                        with open(absfile, mode="rb") as f_read:
                            lines = f_read.readlines()
                            for line in lines:
                                arr.append(list(re.findall(reg, line)[0]))
                        if arr:
                            df = pd.DataFrame(arr)

                            df[1] = pd.to_numeric(df[1], errors='ignore')
                            upper_bound = 500
                            df = df[df[1].__le__(upper_bound)]

                            max_arg = pd.to_numeric(df[0], errors='ignore').argmax()
                            max_val, max_num, max_type, max_counter = df[0][max_arg], df[1][max_arg], df[2][
                                max_arg], df[1].max()
                            filename = 'test.txt'
                            if train:
                                filename = 'train.txt'

                            with open('%s/%s' % (output, filename), mode='a') as f_write:
                                splitted_path = path.split('/')
                                name_val = '%s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
                                    splitted_path[6], splitted_path[7], splitted_path[8], splitted_path[9], max_val,
                                    max_num, max_type, f, max_counter)
                                f_write.writelines(name_val)
                                # create_tsne(train, path, max_num)
                    except Exception as e:
                        print e.message


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='an integer for the accumulator')
    parser.add_argument('--output', type=str, help='an integer for the accumulator')
    args = parser.parse_args()
    try:
        os.remove('%s/test.txt' % args.output)
    except:
        pass
    try:
        os.remove('%s/train.txt' % args.output)
    except:
        pass
    find_best(args.path, args.output)
    find_best(args.path, args.output, train=True)
