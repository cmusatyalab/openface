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
import imagehash
from PIL import Image

import faceapi
from faceapi import FaceCenter
from faceapi.utils import log_center
# from faceapi import exceptions


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
    def __init__(self, db_file, trained_face_dir):
        super(FaceCenterOf, self).__init__(db_file, trained_face_dir)
        self._log = log_center.make_logger(__name__, logging.DEBUG)

        self._trained_face_dir = trained_face_dir
        if not os.path.exists(trained_face_dir):
            os.makedirs(trained_face_dir)

        self._save_result = False
        self._face_db = faceapi.database.make_db_manager(db_file)
        self._face_detector = faceapi.detecter.make_detector()
        self._face_eigener = faceapi.eigener.make_eigener()
        self._face_classifier = faceapi.classifier.make_classifier(db_file)
        self._training_id = -1
        self._trainint_name = ''
        self._trainint_cb = None

    def faceList(self):
        list = []
        for info in self._face_db.dbList():
            face_info = faceapi.FaceInfo(
                info['hash'].encode('ascii', 'ignore'),
                info['name'],
                [float(x) for x in info['eigen'].split(',')],
                info['src_hash'].encode('ascii', 'ignore'),
                info['face_img'].encode('ascii', 'ignore'),
                info['class_id'])
            list.append(face_info)
        return list

    # def train(self, image, name):
    #     if isinstance(image, basestring):
    #         img = Image.open(image).convert('RGB')
    #         self._logger.debug("PIL image: {}".format(str(img)))
    #     elif isinstance(image, Image.Image):
    #         img = image.convert('RGB')
    #     else:
    #         raise exceptions.LibError("Unknow image type")

    #     src_hash = str(imagehash.phash(img, hash_size=16))

    #     identity = self._toIdentity(name)
    #     if identity is None:
    #         people_list = self._face_db.distinct_search(
    #                                     ['name', 'class_id'], 'class_id')
    #         identity = len(people_list)

    #     bbs = self._face_detector.detect(img)

    #     trained_list = []
    #     for face in bbs:
    #         phash, rep = self._face_eigener.eigenValue(face.img)

    #         face_img = os.path.join(
    #                 self._trained_face_dir, name, "{}.jpg".format(phash))
    #         Image.fromarray(face.img).save(face_img)
    #         record = faceapi.FaceInfo(
    #                     phash, name, rep, src_hash, face_img, identity)
    #         self._face_db.addList([record])
    #         trained_list.append(record)
    #         # content = [str(x) for x in face.img.flatten()]

    #     self._face_classifier.updateDB()
    #     return trained_list

    def start_train(self, name, cb=None):
        self._training_id = self._toIdentity(name)
        self._trainint_name = name
        self._trainint_cb = cb

        if self._training_id is None:
            people_list = self._face_db.distinct_search(
                ['name', 'class_id'], 'class_id')

            self._training_id = len(people_list)

        save_dir = os.path.join(self._trained_face_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train(self, img):
        if self._training_id == -1:
            self._log.error('[ERROR] Call start_train first, please.')
            return False

        add_face_list = []
        # for img in image_list:
        self._log.debug("training img: {}".format(img))
        pil_img = self._toPilImg(img)
        src_hash = str(imagehash.phash(pil_img, hash_size=16))
        db_list = self._face_db.search(
            'src_hash',
            '\'{}\''.format(src_hash),
            1)

        if len(db_list) > 0:
            self._log.debug('trained image, skip it')
            return False

        # start to train the faces in the image
        t = time.time()
        bbs = self._face_detector.detect(img)
        t = time.time() - t
        self._log.debug("face detection done({})".format(t))
        t = time.time()

        record = None
        for face in bbs:
            # every single face in a image
            phash, rep = self._face_eigener.eigenValue(face.img)

            save_dir = os.path.join(
                self._trained_face_dir,
                self._trainint_name)

            face_img = os.path.join(save_dir, "{}.jpg".format(phash))
            Image.fromarray(face.img).save(face_img)

            record = faceapi.FaceInfo(
                phash,
                self._trainint_name,
                rep,
                src_hash,
                face_img,
                self._training_id)

            add_face_list.append(record)
            if self._trainint_cb is not None:
                self._trainint_cb(face.area, face.landmarks)

        t = time.time() - t
        self._log.debug("training img done({})".format(t))

        self._face_db.addList(add_face_list)
        return record

    def finish_train(self):
        if self._training_id == -1:
            self._log.error('[ERROR] Call start_train first, please.')
            return

        self._face_classifier.updateDB()
        self._training_id = -1
        self._trainint_name = ''
        self._trainint_cb = None

    def predict(self, image, callback=None):
        t = time.time()
        bbs = self._face_detector.detect(image)
        self._log.info('face detectime: {}'.format(time.time() - t))
        hit_cnt = 0
        t = time.time()
        for face in bbs:
            hit = self._face_classifier.predict(face.img)
            phash, rep = self._face_eigener.eigenValue(face.img)
            if hit is None:
                continue
            hit_cnt += 1
            if callback is not None:
                callback(
                    hit.class_id, hit.name, face.area,
                    face.landmarks, hit.scroe)

            # print 'hit name: {}'.format(hit.name)
            if self._save_result:
                path = '/Users/cowbjt/project/FaceRecognition/filtered_result'
                hit_dir = os.path.join(path, hit.name)
                if not os.path.isdir(hit_dir):
                    os.makedirs(hit_dir)
                fname = os.path.join(hit_dir, '{}.jpg'.format(phash))
                Image.fromarray(face.img).save(fname)

        self._log.info('took time: {}'.format(time.time() - t))
        return hit_cnt

    def trainDir(self, dir_path):
        if not os.path.isdir(dir_path):
            self._log.error('Not a dir, do nothing.\n({})'.format(dir_path))
            return

        trian_names = next(os.walk(dir_path))[1]
        for name in trian_names:
            self.start_train(name)
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

            t = time.time()
            self._log.info(
                "train >>>>> name: {}, svm id: {}".format(name, class_id))

            exts = ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
            train_imgs = []
            for ext in exts:
                train_imgs.extend(glob.glob(os.path.join(path, ext)))

            for img in train_imgs:
                self.train(img)

            t = time.time() - t
            self._log.info(
                "<<<({}) end name: {}, svm id: {}".format(t, name, class_id))

        self.finish_train()
        # self._face_classifier.updateDB()

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

    def _toPilImg(self, image):
        if isinstance(image, basestring):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            img = None

        return img
