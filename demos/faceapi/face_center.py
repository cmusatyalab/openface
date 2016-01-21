# -*- coding: UTF-8 -*-

"""
@file face_center.py
@brief
    Implement of FaceCenter.

Created on: 2016/1/21
"""

import os
import glob
import logging
import time
# import imagehash
# from PIL import Image

import faceapi
from faceapi import FaceCenter
from faceapi.utils import log_center


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


class FaceCenterOf(FaceCenter):
    def __init__(self, db_path):
        super(FaceCenterOf, self).__init__(db_path)
        self._log = log_center.make_logger(__name__, logging.DEBUG)

        self._face_db = faceapi.database.make_db_manager(db_path)
        self._face_detector = faceapi.detecter.make_detector()
        self._face_eigener = faceapi.eigener.make_eigener()
        self._face_classifier = faceapi.classifier.make_classifier(db_path)

    def faceList(self):
        list = []
        for info in self._face_db.dbList():
            face_info = faceapi.FaceInfo(
                                info['hash'].encode('ascii', 'ignore'),
                                info['name'],
                                [float(x) for x in info['eigen'].split(',')],
                                info['img_path'].encode('ascii', 'ignore'),
                                info['class_id'])
            list.append(face_info)
        return list

    def train(self, image, name):
        bbs = self._face_detector.detect(image)

        trained_list = []
        for face in bbs:
            phash, rep = self._face_eigener.eigenValue(face.img)
            identity = self._toIdentity(name)
            if identity is None:
                people_list = self._face_db.distinct_search(
                                            ['name', 'class_id'], 'class_id')

                identity = len(people_list)

            record = faceapi.FaceInfo(
                             phash, name, rep, "./test.png", identity)
            self._face_db.addList([record])
            trained_list.append(record)
            # content = [str(x) for x in face.img.flatten()]

        self._face_classifier.updateDB()
        return trained_list

    def predict(self, image, callback=None):
        bbs = self._face_detector.detect(image)
        hit_cnt = 0
        for face in bbs:
            hit = self._face_classifier.predict(face.img)
            if hit is None:
                continue
            hit_cnt += 1
            if callback is not None:
                callback(hit.class_id, face.area, face.landmarks)

        return hit_cnt

    def trainDir(self, dir_path):
        if not os.path.isdir(dir_path):
            self._log.error('Not a dir, do nothing.\n({})'.format(dir_path))
            return

        trian_names = next(os.walk(dir_path))[1]
        for name in trian_names:
            path = os.path.join(dir_path, name)
            self._log.debug("Going to train: {}".format(path))

            # check if this person exist
            db_names = self._face_db.distinct_search(
                                        ['name', 'class_id'], 'class_id')
            self._log.debug("db_names: {}".format(db_names))
            check_ret = [
                        (name_dic['name'], name_dic['class_id'])
                        for name_dic in db_names
                        if name_dic['name'] == name]

            self._log.debug("check_ret: {}".format(check_ret))
            class_id = len(db_names)
            if len(check_ret) > 0:
                class_id = (check_ret[0])[1]

            tp = time.time()
            self._log.info(
                    "train >>>>> name: {}, svm id: {}".format(name, class_id))

            exts = ("*.png", "*.jpg", "*.jpeg", "JPG")
            train_imgs = []
            for ext in exts:
                train_imgs.extend(glob.glob(os.path.join(path, ext)))

            # # print "train imgs: {}".format(train_imgs)
            for img in train_imgs:
                self._log.debug("training img: {}".format(img))
                # one training image for a person
                t = time.time()
                bbs = self._face_detector.detect(img)
                t = time.time() - t
                self._log.debug("face detection done({})".format(t))
                t = time.time()
                for face in bbs:
                    # every single face in a image
                    phash, rep = self._face_eigener.eigenValue(face.img)
                    record = faceapi.FaceInfo(
                                 phash, name, rep, "./test.png", class_id)
                    self._face_db.addList([record])
                t = time.time() - t
                self._log.debug("face training done({})".format(t))

            tp = time.time() - tp
            self._log.info(
                "<<<({}) end name: {}, svm id: {}".format(tp, name, class_id))

        self._face_classifier.updateDB()

    def _toIdentity(self, name):
        db_name_map = self._face_db.distinct_search(
                                            ['name', 'class_id'], 'class_id')

        if len(db_name_map) == 0:
            return None

        check_ret = [
                    (name_dic['name'], name_dic['class_id'])
                    for name_dic in db_name_map
                    if name_dic['name'] == name]

        if len(check_ret) == 0:
            return None

        class_id = (check_ret[0])[1]

        return class_id
