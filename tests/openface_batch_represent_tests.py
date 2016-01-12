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
import tempfile

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

    workDir = tempfile.mkdtemp(prefix='OpenFaceBatchRep-')

    cmd = ['python2', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(workDir, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['th', './batch-represent/main.lua',
           '-data', os.path.join(workDir, 'aligned'),
           '-outDir', os.path.join(workDir, 'reps')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    fname = os.path.join(workDir, 'reps', 'labels.csv')
    labels = pd.read_csv(fname, header=None).as_matrix()
    fname = os.path.join(workDir, 'reps', 'reps.csv')
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    brody1 = brody2 = None
    for i, (cls, label) in enumerate(labels):
        if "Brody_0001" in label:
            brody1 = embeddings[i]
        elif "Brody_0002" in label:
            brody2 = embeddings[i]

    assert brody1 is not None
    assert brody2 is not None
    print("brody1:", brody1)
    print("brody2:", brody2)

    cosDist = scipy.spatial.distance.cosine(brody1, brody2)
    print('cosDist:', cosDist)
    assert np.isclose(cosDist, 0.15684134)

    shutil.rmtree(workDir)
