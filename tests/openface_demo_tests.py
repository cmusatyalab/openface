# OpenFace tests, run with `nosetests-2.7 -v -d test.py`
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
import re
import shutil

from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
modelDir = os.path.join(openfaceDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

exampleImages = os.path.join(openfaceDir, 'images', 'examples')
lfwSubset = os.path.join(openfaceDir, 'data', 'lfw-subset')

dlibFacePredictor = os.path.join(dlibModelDir,
                                 "shape_predictor_68_face_landmarks.dat")
nn4_v1_model = os.path.join(openfaceModelDir, 'nn4.v1.t7')
nn4_v2_model = os.path.join(openfaceModelDir, 'nn4.v2.t7')


def test_compare_demo():
    cmd = ['python2', os.path.join(openfaceDir, 'demos', 'compare.py'),
           os.path.join(exampleImages, 'lennon-1.jpg'),
           os.path.join(exampleImages, 'lennon-2.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out, err)
    assert "0.463" in out


def test_classification_demo_pretrained():
    cmd = ['python2', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'infer',
           os.path.join(openfaceDir, 'models', 'openface',
                        'celeb-classifier.nn4.v2.pkl'),
           os.path.join(exampleImages, 'carell.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out, err)
    assert "Predict SteveCarell with 0.89 confidence." in out


def test_classification_demo_training():
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

    cmd = ['python2', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'train',
           os.path.join(lfwSubset, 'reps')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    assert p.returncode == 0

    cmd = ['python2', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'infer',
           os.path.join(lfwSubset, 'reps', 'classifier.pkl'),
           os.path.join(lfwSubset, 'raw', 'Adrien_Brody', 'Adrien_Brody_0001.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    print(out, err)
    m = re.search('Predict (.*) with (.*) confidence', out)
    assert m is not None
    assert m.group(1) == 'Adrien_Brody'
    assert float(m.group(2)) >= 0.80

    shutil.rmtree(os.path.join(lfwSubset, 'aligned'))
    shutil.rmtree(os.path.join(lfwSubset, 'reps'))
