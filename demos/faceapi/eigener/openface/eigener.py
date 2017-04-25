# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""

# import os
# import glob
import numpy as np
import logging
import imagehash
from PIL import Image

# import faceapi
from faceapi import openfaceutils
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.eigener import ImgEigener

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


class EigenerOf(ImgEigener):
    def __init__(self):
        super(EigenerOf, self).__init__()
        self._log = log_center.make_logger(__name__, logging.DEBUG)
        # self._face_db = faceapi.database.make_db_manager()

    def eigenValue(self, image):
        if isinstance(image, basestring):
            pil_img = Image.open(image)
            np_img = np.asarray(pil_img)
            self._log.debug("PIL image: {}".format(str(pil_img)))
        elif isinstance(image, Image.Image):
            pil_img = image
            np_img = np.asarray(image)
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
            np_img = image
        else:
            raise exceptions.LibError("Unknow know img type")

        rep = openfaceutils.neural_net.forward(np_img)
        # self._log.info("eigen: \n{}".format(rep))

        # rep_str = ",".join(
        #                 str(evalue)
        #                 for evalue in rep.tolist())

        phash = str(imagehash.phash(pil_img))

        # record = faceapi.FaceInfo(
        #                     phash, name, rep_str, "./test.png", class_id)
        #
        # self._face_db.addList([record])

        return (phash, rep.tolist())

    # def trainDir(self, dir_path):
    #     dir_path = '/openface/Develop/openface/demos/web/train_img'
    #     if not os.path.isdir(dir_path):
    #         self._log.error('Not a dir, do nothing.\n({})'.format(dir_path))
    #         return

    #     db_names = self._face_db.distinct_search(
    #                             ['name', 'class_id'], 'class_id')
    #     trian_names = next(os.walk(dir_path))[1]

    #     self._log.debug("db_names: {}".format(db_names))

    #     for name in trian_names:
    #         path = os.path.join(dir_path, name)
    #         self._log.debug("Going to train: {}".format(path))

    #     # check if this person exist
    #     check_ret = [
    #                 (name_dic['name'], name_dic['class_id'])
    #                 for name_dic in db_names
    #                 if name_dic['name'] == name]

    #     self._log.debug("check_ret: {}".format(check_ret))
    #     class_id = len(db_names)
    #     if len(check_ret) > 0:
    #         class_id = (check_ret[0])[1]

    #     self._log.info("train >>> name: {}, svm id: {}".format(name, class_id))

    #     exts = ("*.png", "*.jpg", "*.jpeg", "JPG")
    #     train_imgs = []
    #     for ext in exts:
    #         train_imgs.extend(glob.glob(os.path.join(path, ext)))

    #     # # print "train imgs: {}".format(train_imgs)
    #     for img in train_imgs:
    #         print "training img: {}".format(img)
    #         # one training image for a person
    #         bbs = _face_detector.detect(img)
    #         for face in bbs:
    #             # every single face in a image
    #             _face_trainer.train(face.img, name, class_id)

    #     print "<<< end name: {}, svm id: {}".format(name, class_id)
