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

from subprocess import Popen, PIPE
import os.path

myDir = os.path.dirname(os.path.realpath(__file__))

class TorchWrap:
    def __init__(self, model='models/facenet/nn4.v1.t7', imgDim=96, cuda=False):
        cmd = ['/usr/bin/env', 'th', os.path.join(myDir,'facenet_server.lua'),
               '-model', model, '-imgDim', str(imgDim)]
        if cuda:
            cmd.append('-cuda')
        self.p = Popen(cmd, stdin=PIPE, stdout=PIPE, bufsize=0)

    def forward(self, imgPath, timeout=10):
        self.p.stdin.write(imgPath+"\n")
        print([float(x) for x in self.p.stdout.readline().strip().split(',')])
