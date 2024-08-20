#!/usr/bin/env python3
#
# Copyright 2015-2024 Carnegie Mellon University
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

import argparse
import functools
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import openface

SUPPORTED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
REPS_CSV_FILE = 'reps.csv'
LABELS_CSV_FILE = 'labels.csv'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DEFAULT_DLIB_MODEL_PATH = os.path.join(MODEL_DIR, 'dlib', 'shape_predictor_68_face_landmarks.dat')
DEFAULT_OPENFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'openface', 'nn4.small2.v1.pt')
IMG_DIM = 96


class OpenFaceDataset(Dataset):
    def __init__(self, dataset_dir, annotations_file=None, transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        if annotations_file is None:
            class_folders = [sub.name for sub in os.scandir(dataset_dir) if sub.is_dir()]
            img_list = []
            for class_name in class_folders:
                class_path = os.path.join(dataset_dir, class_name)
                for img in os.scandir(class_path):
                    if img.name.lower().split('.')[-1] in SUPPORTED_IMAGE_EXTENSIONS:
                        img_list.append({'filename': os.path.join(class_path, img.name),
                                         'label': class_name})
            self.img_labels = pd.DataFrame(img_list)
        else:
            self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            rgb_img = self.transform(rgb_img)
        if self.target_transform:
            label = self.target_transform(label)
        if rgb_img is None:
            print("Warning: Unable to find a face: {}".format(img_path))
        return rgb_img, label


def preprocess(image):
    if image is None:
        return None
    image = (image / 255.).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # channel-first ordering
    return image


def get_or_add(key, dictionary):
    if key in dictionary:
        return dictionary.get(key)
    else:
        val = len(dictionary) + 1
        dictionary[key] = val
        return val


# Reference:
# https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another/69578320#69578320
def collate_filter_none(batch, dataset):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        print("Got a batch with no valid images. Replacing with a random image from the dataset instead. "
              "If you do not want this behavior, remove those images where faces cannot be detected from "
              "the input directory.")
        batch = [dataset[np.random.randint(low=0, high=len(dataset))]]
        return collate_filter_none(batch, dataset)
    return torch.utils.data.dataloader.default_collate(batch)


def write_reps_and_labels(arg):
    input_dataset_dir = arg.input
    output_csv_dir = arg.output
    reps_csv_path = os.path.join(output_csv_dir, REPS_CSV_FILE)
    labels_csv_path = os.path.join(output_csv_dir, LABELS_CSV_FILE)
    for csv_path in [reps_csv_path, labels_csv_path]:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    os.makedirs(output_csv_dir, exist_ok=True)

    label_dict = {}
    label_counter = Counter()
    if arg.aligned:
        dataset = OpenFaceDataset(input_dataset_dir, transform=preprocess)
    else:
        align = openface.AlignDlib(arg.dlib_model_path)
        landmark_map = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if arg.landmarks not in landmark_map:
            raise Exception("Landmarks unrecognized: {}".format(arg.landmarks))
        landmark_indices = landmark_map[arg.landmarks]
        def align_and_preprocess(image):
            image = align.align(IMG_DIM, image, landmarkIndices=landmark_indices)
            return preprocess(image)
        dataset = OpenFaceDataset(input_dataset_dir, transform=align_and_preprocess)

    dataloader = DataLoader(dataset, batch_size=arg.batch, shuffle=arg.shuffle, num_workers=arg.worker,
                            collate_fn=functools.partial(collate_filter_none, dataset=dataset))

    model = openface.OpenFaceNet()
    if arg.cpu:
        model.load_state_dict(torch.load(arg.openface_model_path))
    else:
        model.load_state_dict(torch.load(arg.openface_model_path, map_location='cuda'))
        model.to(torch.device('cuda'))
    model.eval()

    for step, (images, labels) in enumerate(dataloader):
        print("Generating representations for batch {}/{} ...".format(step, len(dataloader)))
        if not arg.cpu:
            images = images.to(torch.device('cuda'))
        reps = model(images)
        reps = reps.cpu().detach().numpy()

        with open(reps_csv_path, 'a') as reps_file:
            np.savetxt(reps_file, reps, fmt='%.8f', delimiter=',')

        label_counter.update(labels)
        with open(labels_csv_path, 'a') as labels_file:
            for label in labels:
                labels_file.write('{},{}\n'.format(get_or_add(label, label_dict), label))
    print("Summary: {} images in total".format(sum(label_counter.values())))
    print(dict(label_counter))
    print("CSV files saved to: {}".format(arg.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, type=str, help='path to input raw image directory')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to output csv directory')
    parser.add_argument('--dlib_model_path', type=str, default=DEFAULT_DLIB_MODEL_PATH,
                        help='path to dlib model')
    parser.add_argument('--openface_model_path', type=str, default=DEFAULT_OPENFACE_MODEL_PATH,
                        help='path to pretrained openface model')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
    parser.add_argument('--aligned', action='store_true',
                        help='if flag is set, assume input images are already aligned')
    parser.add_argument('--landmarks', type=str, choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                        default='outerEyesAndNose', help='landmarks to align to')
    parser.add_argument('--cpu', action='store_true', help='run model on CPU only')
    args = parser.parse_args()
    write_reps_and_labels(args)
