#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                self.trainSVM()
        elif msg['type'] == "ADD_PERSON":
            self.people.append(msg['val'].encode('ascii', 'ignore'))
            print(self.people)
        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'])
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        if not training:
            self.trainSVM()

    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def sendTSNE(self, people):
        d = self.getData()
        if d is None:
            return
        else:
            (X, y) = d

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        yVals = list(np.unique(y))
        colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

        # print(yVals)

        plt.figure()
        for c, i in zip(colors, yVals):
            name = "Unknown" if i == -1 else people[i]
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
            plt.legend()

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)

        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "TSNE_DATA",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

    def trainSVM(self):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1:
                return

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    def processFrame(self, dataURL, identity):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        if not self.training:
            annotatedFrame = np.copy(buf)

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        identities = []
        # bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            # print(len(bbs))
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue

            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            if phash in self.images:
                identity = self.images[phash].identity
            else:
                rep = net.forward(alignedFace)
                # print(rep)
                if self.training:
                    self.images[phash] = Face(rep, identity)
                    # TODO: Transferring as a string is suboptimal.
                    # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                    # fx=0.5, fy=0.5).flatten()]
                    content = [str(x) for x in alignedFace.flatten()]
                    msg = {
                        "type": "NEW_IMAGE",
                        "hash": phash,
                        "content": content,
                        "identity": identity,
                        "representation": rep.tolist()
                    }
                    self.sendMessage(json.dumps(msg))
                else:
                    if len(self.people) == 0:
                        identity = -1
                    elif len(self.people) == 1:
                        identity = 0
                    elif self.svm:
                        identity = self.svm.predict(rep)[0]
                    else:
                        print("hhh")
                        identity = -1
                    if identity not in identities:
                        identities.append(identity)

            if not self.training:
                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=3)
                for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                    cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                               color=(102, 204, 255), thickness=-1)
                if identity == -1:
                    if len(self.people) == 1:
                        name = self.people[0]
                    else:
                        name = "Unknown"
                else:
                    name = self.people[identity]
                cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(152, 255, 204), thickness=2)

        if not self.training:
            msg = {
                "type": "IDENTITIES",
                "identities": identities
            }
            self.sendMessage(json.dumps(msg))

            plt.figure()
            plt.imshow(annotatedFrame)
            plt.xticks([])
            plt.yticks([])

            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, format='png')
            imgdata.seek(0)
            content = 'data:image/png;base64,' + \
                urllib.quote(base64.b64encode(imgdata.buf))
            msg = {
                "type": "ANNOTATED",
                "content": content
            }
            plt.close()
            self.sendMessage(json.dumps(msg))


def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
