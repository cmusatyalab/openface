# OpenFace training tests.
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
import shutil

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import tempfile

from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
modelDir = os.path.join(openfaceDir, 'models')

exampleImages = os.path.join(openfaceDir, 'images', 'examples')
lfwSubset = os.path.join(openfaceDir, 'data', 'lfw-subset')


def test_dnn_training():
    assert os.path.isdir(
        lfwSubset), "Get lfw-subset by running ./data/download-lfw-subset.sh"

    imgWorkDir = tempfile.mkdtemp(prefix='OpenFaceTrainingTest-Img-')
    cmd = ['python2', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(imgWorkDir, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['python2', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(imgWorkDir, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    netWorkDir = tempfile.mkdtemp(prefix='OpenFaceTrainingTest-Net-')
    saveDir = os.path.join(netWorkDir, '1')
    cmd = ['th', './main.lua',
           '-data', os.path.join(imgWorkDir, 'aligned'),
           '-modelDef', '../models/openface/nn4.def.lua',
           '-peoplePerBatch', '3',
           '-imagesPerPerson', '10',
           '-nEpochs', '10',
           '-epochSize', '1',
           '-cache', netWorkDir,
           '-save', saveDir,
           '-cuda', '-cudnn', '-testing',
           '-nDonkeys', '-1']
    p = Popen(cmd, stdout=PIPE, stderr=PIPE,
              cwd=os.path.join(openfaceDir, 'training'))
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    # Training won't make much progress on lfw-subset, but as a sanity check,
    # make sure the training code runs and doesn't get worse than 0.2.
    trainLoss = pd.read_csv(os.path.join(saveDir, 'train.log'),
                            sep='\t').as_matrix()[:, 0]
    assert np.mean(trainLoss) < 0.3

    shutil.rmtree(imgWorkDir)
    shutil.rmtree(netWorkDir)
