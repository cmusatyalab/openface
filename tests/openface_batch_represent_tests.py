# OpenFace batch-represent tests.
#
# Copyright 2015 Carnegie Mellon University
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
import scipy
import scipy.spatial

from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
modelDir = os.path.join(openfaceDir, 'models')

exampleImages = os.path.join(openfaceDir, 'images', 'examples')
lfwSubset = os.path.join(openfaceDir, 'data', 'lfw-subset')


def test_batch_represent():
    # Get lfw-subset by running ./data/download-lfw-subset.sh
    assert os.path.isdir(lfwSubset)

    cmd = ['python2', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(lfwSubset, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    assert p.returncode == 0

    cmd = ['th', './batch-represent/main.lua',
           '-data', os.path.join(lfwSubset, 'aligned'),
           '-outDir', os.path.join(lfwSubset, 'reps')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    assert p.returncode == 0

    fname = os.path.join(lfwSubset, 'reps', 'labels.csv')
    labels = pd.read_csv(fname, header=None).as_matrix()
    fname = os.path.join(lfwSubset, 'reps', 'reps.csv')
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    brody1 = brody2 = None
    for i, (cls, label) in enumerate(labels):
        if "Brody_0001" in label:
            brody1 = embeddings[i]
        elif "Brody_0002" in label:
            brody2 = embeddings[i]

    assert brody1 is not None
    assert brody2 is not None

    cosDist = scipy.spatial.distance.cosine(brody1, brody2)
    assert np.isclose(cosDist, 0.113500484192)

    shutil.rmtree(os.path.join(lfwSubset, 'aligned'))
    shutil.rmtree(os.path.join(lfwSubset, 'reps'))
