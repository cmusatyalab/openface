# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""

import os
import time
import numpy as np
import logging
import imagehash
from PIL import Image

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib

import faceapi
from faceapi import openfaceutils
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.classifier import FaceClassifier

"""
8888888b.            .d888 d8b
888  "Y88b          d88P"  Y8P
888    888          888
888    888  .d88b.  888888 888 88888b.   .d88b.  .d8888b
888    888 d8P  Y8b 888    888 888 "88b d8P  Y8b 88K
888    888 88888888 888    888 888  888 88888888 "Y8888b.
888  .d88P Y8b.     888    888 888  888 Y8b.          X88
8888888P"   "Y8888  888    888 888  888  "Y8888   88888P'
"""

_DEFAULT_IMG_W = 400
_DEFAULT_IMG_H = 300

_DEFAULT_SVM_FILE_NAME = 'svm.pkl'

"""
.d8888b.  888
d88P  Y88b 888
888    888 888
888        888  8888b.  .d8888b  .d8888b
888        888     "88b 88K      88K
888    888 888 .d888888 "Y8888b. "Y8888b.
Y88b  d88P 888 888  888      X88      X88
 "Y8888P"  888 "Y888888  88888P'  88888P'
 """


class ClassifierOf(FaceClassifier):
    def __init__(self, db_file):
        super(ClassifierOf, self).__init__(db_file)
        self._log = log_center.make_logger(__name__, logging.INFO)
        self._face_db = faceapi.database.make_db_manager(db_file)

        db_dir = os.path.dirname(os.path.realpath(db_file))
        svm_dir = os.path.join(db_dir, 'svm')
        self._svm_file = os.path.join(svm_dir, _DEFAULT_SVM_FILE_NAME)

        if not os.path.exists(svm_dir):
            os.makedirs(svm_dir)

        self._db_dict = {}
        for info in self._face_db.dbList():
            h = info['hash'].encode('ascii', 'ignore')
            info.pop("hash", None)
            self._db_dict[h] = info

        if os.path.isfile(self._svm_file):
            self._log.info("Load saved svm result")
            self._svm = joblib.load(self._svm_file)
        else:
            self._log.info("No saved svm, update from db")
            self.updateDB()

    def updateDB(self):
        self._db_dict = {}
        for info in self._face_db.dbList():
            h = info['hash'].encode('ascii', 'ignore')
            info.pop("hash", None)
            self._db_dict[h] = info
        # self._log.info("_db_dict: {}".format(self._db_dict))

        # train svm
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}]
        d = self._trainData()
        if d is None:
            self._svm = None
            return

        self._log.info('Training svm')
        t = time.time()
        (X, y) = d
        self._svm = GridSearchCV(
            SVC(C=1, probability=True),
            param_grid, cv=5).fit(X, y)
        # self._log.info("train svm: {}".format(self._svm))

        joblib.dump(self._svm, self._svm_file)
        # print 'save svm: {}'.format(ret)
        self._log.info('Training svm done({})'.format(time.time() - t))

    def predict(self, image):
        if self._svm is None:
            self._log.warn("self._svm is None")
            return None
        elif len(self._db_dict) == 0:
            self._log.warn("self._db_dict == 0")
            return None

        if isinstance(image, basestring):
            pil_img = Image.open(image)
            np_img = np.asarray(pil_img)
            self._log.debug("PIL image: {}".format(str(pil_img)))
        elif isinstance(image, Image.Image):
            pil_img = image
            np_img = np.asarray(image)
        elif isinstance(image, np.ndarray):
            np_img = image
            pil_img = Image.fromarray(image)
        else:
            raise exceptions.LibError("Unknow image type")

        resultRecord = None
        phash = str(imagehash.phash(pil_img))
        if phash in self._db_dict:
            hit = self._db_dict[phash]
            hit_p = 0.95
            self._log.debug('match with hash: {}'.format(hit['name']))
        else:
            rep = openfaceutils.neural_net.forward(np_img)
            # predict_ret = self._svm.predict(rep)
            predict_p = self._svm.predict_proba(rep)[0]
            hit_p = max(predict_p)
            # if hit_p < (2.0/len(predict_p)):
            #     # probability of hit < average * 2
            #     return None
            # class_id = predict_p.index(hit_p)
            class_id = np.argmax(predict_p)
            self._log.info("svm({}, {}):\n{}".format(
                class_id, predict_p.max(), predict_p))
            db_list = self._face_db.search('class_id', class_id, 1)
            self._log.debug("result({}): {}".format(len(db_list), db_list))
            hit = db_list[0]

        resultRecord = faceapi.FaceInfo(
            phash,
            hit['name'],
            hit['eigen'],
            'src_img_no_need',
            hit['face_img'],
            hit['class_id'])
        resultRecord.scroe = hit_p
        return resultRecord

    def _trainData(self):
        X = []
        y = []

        # db_list = self._db_dict.values()
        # db_list.sort(key=lambda info: info['class_id'])
        for info in self._db_dict.values():
            rep_list = [float(x) for x in info['eigen'].split(',')]
            X.append(rep_list)
            y.append(info['class_id'])

        db_names = self._face_db.distinct_search(
            ['name', 'class_id'], 'class_id')
        if len(db_names) == 1:
            self._log.info("just one class, do not train svm.")
            return None

        cnt = len(set(y + [-1])) - 1
        if cnt == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)

        self._log.info("classes({}): {}".format(len(y), y))
        # print 'X({}):\n{}'.format(len(X), X)
        return (X, y)
