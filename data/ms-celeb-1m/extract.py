#!/usr/bin/env python3
#
# Copyright 2016 Carnegie Mellon University
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
#
# This script extracts the MS-Celeb-1M TSV file into images on disk.
# For more information, see http://arxiv.org/abs/1607.08221 and http://msceleb.com
#
# Brandon Amos
# 2016-07-29

import argparse
import base64
import csv
import os
# import magic # Detect image type from buffer contents (disabled, all are jpg)

parser = argparse.ArgumentParser()
parser.add_argument('croppedTSV', type=str)
parser.add_argument('--outputDir', type=str, default='raw')
args = parser.parse_args()

with open(args.croppedTSV, 'r') as tsvF:
    reader = csv.reader(tsvF, delimiter='\t')
    i = 0
    for row in reader:
        MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

        saveDir = os.path.join(args.outputDir, MID)
        savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

        # assert(magic.from_buffer(data) == 'JPEG image data, JFIF standard 1.01')

        os.makedirs(saveDir, exist_ok=True)
        with open(savePath, 'wb') as f:
            f.write(data)

        i += 1

        if i % 1000 == 0:
            print("Extracted {} images.".format(i))
