#!/usr/bin/env python2
# projectS and projectC were written by Gabriele Farina.

import time

start = time.time()

import argparse
import cv2
import os
import dlib

import numpy as np
np.set_printoptions(precision=2)
import openface

from matplotlib import cm

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    return reps


def projectS(rho, theta, z):
    p = np.array([np.sqrt(3.) * rho * (np.cos(theta) + np.sin(theta)) / 2.,
                  z + 1. + rho * (np.cos(theta) - np.sin(theta)) / 2.])
    p += np.array([1.5, 0.5])
    p /= 3.
    return p


def projectC(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    if x == 0 and y == 0:
        theta = 0
    elif x >= 0:
        theta = np.arcsin(y / rho)
    else:
        theta = -np.arcsin(y / rho) + np.pi

    return projectS(rho, theta, z)


def draw(pts=[], clrs=[], cSz=400):
    def toFrame(x):
        return tuple((cSz * x).astype(np.int32))

    cFrame = np.full((cSz, cSz, 3), 255, dtype=np.uint8)

    for z in np.linspace(-1, 1, 9):
        r = np.sqrt(1. - z**2)
        last = None
        for theta in np.linspace(0, 2 * np.pi, 50):
            x = toFrame(projectS(r, theta, z))
            if last is not None:
                cv2.line(cFrame, x, last, color=(0, 0, 0))
            last = x

    for x in np.linspace(-1, 1, 9):
        last = None
        for theta in np.linspace(0, 2 * np.pi, 50):
            r = np.sqrt(1. - x**2)
            z = r * np.sin(theta)
            y = r * np.cos(theta)
            # x = toFrame(projectS(r, theta, z))
            p = toFrame(projectC(x, y, z))
            if last is not None:
                cv2.line(cFrame, p, last, color=(0, 0, 0))
            last = p

    s = 1
    x = toFrame(projectC(-s, 0, 0))
    y = toFrame(projectC(s, 0, 0))
    cv2.line(cFrame, x, y, color=(0, 0, 0), thickness=4)

    x = toFrame(projectC(0, -s, 0))
    y = toFrame(projectC(0, s, 0))
    cv2.line(cFrame, x, y, color=(0, 0, 0), thickness=4)

    x = toFrame(projectC(0, 0, -s))
    y = toFrame(projectC(0, 0, s))
    cv2.line(cFrame, x, y, color=(0, 0, 0), thickness=4)

    for pt, c in zip(pts, clrs):
        fPt = toFrame(projectC(pt[0], pt[1], pt[2]))
        fPt_noz = toFrame(projectC(pt[0], pt[1], 0))
        fPt_nozy = toFrame(projectC(pt[0], 0, 0))
        fPt_nozx = toFrame(projectC(0, pt[1], 0))
        cv2.line(cFrame, fPt, fPt_noz, color=c, thickness=2)
        cv2.line(cFrame, fPt_noz, fPt_nozy, color=c, thickness=2)
        cv2.line(cFrame, fPt_noz, fPt_nozx, color=c, thickness=2)
        cv2.circle(cFrame, fPt, 5, color=c, thickness=-1)

    return cFrame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default='nn4.small2.3d.v1.t7')
    # Download the 3D model from:
    # https://storage.cmusatyalab.org/openface-models/nn4.small2.3d.v1.t7
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    # parser.add_argument('--width', type=int, default=640)
    # parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--scale', type=int, default=0.25)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)

    class Tracker:

        def __init__(self, img, bb, rep):
            self.t = dlib.correlation_tracker()
            self.t.start_track(img, bb)
            self.rep = rep
            self.bb = bb
            self.pings = 0

        def updateRep(self, rep):
            self.pings = 0
            alpha = 0.9
            self.rep = alpha * self.rep + (1. - alpha) * rep
            return self.rep

        def overlap(self, bb):
            p = float(self.bb.intersect(bb).area()) / float(self.bb.area())
            return p > 0.3

        def ping(self):
            self.pings += 1

    trackers = []

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        frameSmall = cv2.resize(frame, (int(args.width * args.scale),
                                        int(args.height * args.scale)))

        bbs = align.getAllFaceBoundingBoxes(frameSmall)

        pts, clrs = [], []
        for i, bb in enumerate(bbs):
            alignedFace = align.align(96, frameSmall, bb,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            rep = net.forward(alignedFace)

            center = bb.center()
            centerI = 0.7 * center.x * center.y / \
                (args.scale * args.scale * args.width * args.height)
            color_np = cm.Set1(centerI)
            color_cv = list(np.multiply(color_np[:3], 255))

            bl = (int(bb.left() / args.scale), int(bb.bottom() / args.scale))
            tr = (int(bb.right() / args.scale), int(bb.top() / args.scale))
            cv2.rectangle(frame, bl, tr, color=color_cv, thickness=3)

            tracked = False
            for i in xrange(len(trackers) - 1, -1, -1):
                t = trackers[i]
                t.t.update(frame)
                if t.overlap(bb):
                    rep = t.updateRep(rep)
                    pts.append(rep)
                    clrs.append(color_cv)
                    tracked = True
                    break

            if not tracked:
                trackers.append(Tracker(frame, bb, rep))
                pts.append(rep)
                clrs.append(color_cv)

        for i in xrange(len(trackers) - 1, -1, -1):
            t = trackers[i]
            t.ping()
            if t.pings > 10:
                del trackers[i]
                continue

            for j in range(i):
                if t.t.get_position().intersect(trackers[j].t.get_position()).area() / \
                   t.t.get_position().area() > 0.4:
                    del trackers[i]
                    continue

        cSz = 450
        sphere = np.copy(frame)
        sphere[0:cSz, 0:cSz, :] = draw(pts, clrs, cSz)
        alpha = 0.25
        beta = 1. - alpha
        cv2.putText(sphere, "CMU OpenFace", (50, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.,
                    (0, 0, 0), 1, cv2.cv.CV_AA)
        cv2.addWeighted(frame, alpha, sphere, beta, 0.0, frame)
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
