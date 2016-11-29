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
#
# This file is Vitalius Parubochyi's modification of `torch_neural_net.py`
# to use lutorpy instead of calling a Lua subprocess.
# It's currently not used by default to avoid adding an
# additional dependency.
# More details are available on this mailing list thread:
# https://groups.google.com/forum/#!topic/cmu-openface/Jj68LJBdN-Y

"""Module for Torch-based neural network usage."""

import lutorpy as lua
import numpy as np
import binascii
import cv2
import os

torch = lua.require('torch')
nn = lua.require('nn')
dpnn = lua.require('dpnn')
image = lua.require('image')


myDir = os.path.dirname(os.path.realpath(__file__))


class TorchNeuralNet:
    """Use a `Torch <http://torch.ch>` and `Lutorpy <https://github.com/imodpasteur/lutorpy>`."""

    #: The default Torch model to use.
    defaultModel = os.path.join(
        myDir, '..', 'models', 'openface', 'nn4.small2.v1.t7')

    def __init__(self, model=defaultModel, imgDim=96, cuda=False):
        """__init__(self, model=defaultModel, imgDim=96, cuda=False)

        Instantiate a 'TorchNeuralNet' object.

        :param model: The path to the Torch model to use.
        :type model: str
        :param imgDim: The edge length of the square input image.
        :type imgDim: int
        :param cuda: Flag to use CUDA in the subprocess.
        :type cuda: bool
        """
        assert model is not None
        assert imgDim is not None
        assert cuda is not None

        torch.setdefaulttensortype('torch.FloatTensor')
        self._net = torch.load(model)
        self._net.evaluate(self._net)

        self._tensor = torch.Tensor(1, 3, imgDim, imgDim)
        self._cuda_tensor = None
        if cuda:
            lua.require('cutorch')
            lua.require('cunn')
            self._net = self._net._cuda()
            self._cuda_tensor = torch.CudaTensor(1, 3, imgDim, imgDim)
        self._cuda = cuda
        self._imgDim = imgDim

    def forwardPath(self, imgPath):
        """
        Perform a forward network pass of an image on disk.

        :param imgPath: The path to the image.
        :type imgPath: str
        :return: Vector of features extracted with the neural network.
        :rtype: numpy.ndarray
        """
        assert imgPath is not None

        self._tensor[0] = image.load(imgPath, 3, 'float')
        self._tensor[0] = image.scale(
            self._tensor[0], self._imgDim, self._imgDim)
        if self._cuda:
            self._cuda_tensor._copy(self._tensor)
            rep = self._net._forward(self._cuda_tensor)._float()
        else:
            rep = self._net.forward(self._net, self._tensor)
        return rep.asNumpyArray().astype(np.float64)

    def forward(self, rgbImg):
        """
        Perform a forward network pass of an RGB image.

        :param rgbImg: RGB image to process. Shape: (imgDim, imgDim, 3)
        :type rgbImg: numpy.ndarray
        :return: Vector of features extracted from the neural network.
        :rtype: numpy.ndarray
        """
        assert rgbImg is not None

        t = '/tmp/openface-torchwrap-{}.png'.format(
            binascii.b2a_hex(os.urandom(8)))
        bgrImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(t, bgrImg)
        rep = self.forwardPath(t)
        os.remove(t)
        return rep
