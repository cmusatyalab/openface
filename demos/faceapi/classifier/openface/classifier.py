# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""

# import os
import numpy as np
import logging
import imagehash
from PIL import Image

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

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
    def __init__(self, dir_path):
        super(ClassifierOf, self).__init__(dir_path)
        self._log = log_center.make_logger(__name__, logging.INFO)
        self._face_db = faceapi.database.make_db_manager(dir_path)
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
                 'kernel': ['rbf']}
        ]
        d = self._trainData()
        if d is None:
            self._svm = None
            return

        (X, y) = d
        self._svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)
        # self._log.info("train svm: {}".format(self._svm))

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
        else:
            rep = openfaceutils.neural_net.forward(np_img)
            class_id = self._svm.predict(rep)[0]
            db_list = self._face_db.search('class_id', class_id, 1)
            self._log.debug("result({}): {}".format(len(db_list), db_list))
            hit = db_list[0]

        resultRecord = faceapi.FaceInfo(
                                        phash,
                                        hit['name'],
                                        hit['eigen'],
                                        hit['img_path'],
                                        hit['class_id'])
        return resultRecord

    def _trainData(self):
        X = []
        y = []

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

        # self._log.info("classes({}): {}".format(len(y), y))
        return (X, y)
