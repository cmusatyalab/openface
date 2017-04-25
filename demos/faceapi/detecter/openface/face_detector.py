# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""

import numpy as np
import logging
from PIL import Image

import openface
import dlib

from faceapi import openfaceutils
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.detecter import FaceDetector
from faceapi.detecter import FaceDetected

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

_IMG_RESIZE_BASE = 725.0


def _resize(img):
    im_width, im_height = img.size
    ratio = 1.0

    if max(im_width, im_height) < _IMG_RESIZE_BASE:
        return ratio, img

    if im_width >= im_height:
        # resize base on width
        ratio = _IMG_RESIZE_BASE / im_width
        im_height = int(ratio * im_height)
        im_width = int(_IMG_RESIZE_BASE)
    else:
        # resize base on height
        ratio = _IMG_RESIZE_BASE / im_height
        im_width = int(ratio * im_width)
        im_height = int(_IMG_RESIZE_BASE)

    img = img.resize((im_width, im_height), Image.BILINEAR)
    return ratio, img


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


class FaceDetectorOf(FaceDetector):
    def __init__(self):
        super(FaceDetectorOf, self).__init__()
        self._logger = log_center.make_logger(__name__, logging.INFO)

    def detect(self, image):
        if isinstance(image, basestring):
            img = Image.open(image).convert('RGB')
            self._logger.debug("PIL image: {}".format(str(img)))
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise exceptions.LibError("Unknow image type")

        scale, img = _resize(img)
        # buf = np.fliplr(np.asarray(img))
        buf = np.asarray(img)
        buf = np.fliplr(buf)
        im_width, im_height = img.size
        rgbFrame = np.zeros(
            # (_DEFAULT_IMG_H, _DEFAULT_IMG_W, 3),
            (im_height, im_width, 3),
            dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 0]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 2]

        # face_box = openfaceutils.align.getLargestFaceBoundingBox(rgbFrame)
        # face_list = [face_box] if face_box is not None else []
        all_face_list = openfaceutils.align.getAllFaceBoundingBoxes(rgbFrame)
        all_face_list = sorted(
            all_face_list,
            key=lambda rect: rect.width() * rect.height(),
            reverse=True)

        face_list = []
        for x in xrange(0, min(3, len(all_face_list))):
            face_list.append(all_face_list[x])

        detected_list = []
        for face_box in face_list:

            landmarks = openfaceutils.align.findLandmarks(rgbFrame, face_box)

            alignedFace = openfaceutils.align.align(
                openfaceutils.args.imgDim, rgbFrame, face_box,
                landmarks=landmarks,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue

            face = FaceDetected()
            face.img = alignedFace
            face.area = dlib.rectangle(
                left=int(face_box.left() / scale),
                top=int(face_box.top() / scale),
                right=int(face_box.right() / scale),
                bottom=int(face_box.bottom() / scale))

            face.landmarks = []
            for p in landmarks:
                face.landmarks.append((int(p[0] / scale), int(p[1] / scale)))

            detected_list.append(face)

        return detected_list
