import os
from abc import ABCMeta, abstractmethod

from faceapi import database
from faceapi import detecter
from faceapi import eigener
from faceapi import classifier

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


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class FaceInfo:
    def __init__(self, hash, name, eigen, src_hash, face_img, class_id):
        self.hash = hash  # string
        self.name = name  # string
        self.eigen = eigen  # float list
        self.src_hash = src_hash  # string
        self.face_img = face_img  # string
        self.class_id = class_id  # int


class FaceCenter():
    __metaclass__ = ABCMeta

    def __init__(self, db_path, trained_face_dir):
        pass

    @abstractmethod
    def faceList(self):
        "Return a list of FaceInfo in database"
        pass

    @abstractmethod
    def start_train(self, name, cb=None):
        pass

    @abstractmethod
    def train(self, image):
        pass

    @abstractmethod
    def finish_train(self):
        pass

    @abstractmethod
    def trainDir(self, dir_path):
        pass

    @abstractmethod
    def predict(self, image, callback):
        pass


"""
8888888888                   888
888                          888
888                          888
8888888     8888b.   .d8888b 888888  .d88b.  888d888 888  888
888            "88b d88P"    888    d88""88b 888P"   888  888
888        .d888888 888      888    888  888 888     888  888
888        888  888 Y88b.    Y88b.  Y88..88P 888     Y88b 888
888        "Y888888  "Y8888P  "Y888  "Y88P"  888      "Y88888
                                                          888
                                                     Y8b d88P
                                                      "Y88P"
 """


def share_center(db_file, trained_face_dir):
    from faceapi.face_center import FaceCenterOf
    return FaceCenterOf(db_file, trained_face_dir)
