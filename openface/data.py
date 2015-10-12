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

from skimage import io


class Image:

    def __init__(self, cls, name, path):
        self.cls = cls
        self.name = name
        self.path = path
        self.rgb = None

    def getRGB(self, cache=False):
        if self.rgb is not None:
            return self.rgb
        else:
            try:
                rgb = io.imread(self.path)
            except:
                rgb = None
            if cache:
                self.rgb = rgb
            return rgb

    def __repr__(self):
        return "({}, {})".format(self.cls, self.name)


def iterImgs(d):
    exts = [".jpg", ".png"]

    for subdir, dirs, files in os.walk(d):
        for path in files:
            (imageClass, fName) = (os.path.basename(subdir), path)
            (imageName, ext) = os.path.splitext(fName)
            if ext in exts:
                yield Image(imageClass, imageName, os.path.join(subdir, fName))
