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
