# OpenFace demo tests.
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
import re
import shutil
import tempfile

from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exampleImages = os.path.join(openfaceDir, 'images', 'examples')
lfwSubset = os.path.join(openfaceDir, 'data', 'lfw-subset')


def test_compare_demo():
    cmd = ['python3', os.path.join(openfaceDir, 'demos', 'compare.py'),
           os.path.join(exampleImages, 'lennon-1.jpg'),
           os.path.join(exampleImages, 'lennon-2.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert '0.763' in out


def test_classification_demo_pretrained():
    cmd = ['python3', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'infer',
           os.path.join(openfaceDir, 'models', 'openface',
                        'celeb-classifier.nn4.small2.v1.pkl'),
           os.path.join(exampleImages, 'carell.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert 'Predict SteveCarell with 0.97 confidence.' in out


def test_classification_demo_pretrained_multi():
    cmd = ['python3', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'infer', '--multi',
           os.path.join(openfaceDir, 'models', 'openface',
                        'celeb-classifier.nn4.small2.v1.pkl'),
           os.path.join(exampleImages, 'longoria-cooper.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert 'Predict EvaLongoria @ x=91 with 0.99 confidence.' in out
    assert 'Predict BradleyCooper @ x=191 with 0.99 confidence.' in out


def test_classification_demo_training():
    assert os.path.isdir(lfwSubset), 'Get lfw-subset by running ./data/download-lfw-subset.sh'

    workDir = tempfile.mkdtemp(prefix='OpenFaceCls-')

    cmd = ['python3', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(workDir, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['python3', os.path.join(openfaceDir, 'util', 'align-dlib.py'),
           os.path.join(lfwSubset, 'raw'), 'align', 'outerEyesAndNose',
           os.path.join(workDir, 'aligned')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['th', './batch-represent/main.lua',
           '-data', os.path.join(workDir, 'aligned'),
           '-outDir', os.path.join(workDir, 'reps')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['python3', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'train',
           os.path.join(workDir, 'reps')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = ['python3', os.path.join(openfaceDir, 'demos', 'classifier.py'),
           'infer',
           os.path.join(workDir, 'reps', 'classifier.pkl'),
           os.path.join(lfwSubset, 'raw', 'Adrien_Brody', 'Adrien_Brody_0001.jpg')]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    m = re.search('Predict (.*) with (.*) confidence', out)
    assert m is not None
    assert m.group(1) == 'Adrien_Brody'
    assert float(m.group(2)) >= 0.80

    shutil.rmtree(workDir)
