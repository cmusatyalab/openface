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
DEFAULT_DLIB_FACE_PREDICTOR_PATH = os.path.join(MODEL_DIR, 'dlib', 'shape_predictor_68_face_landmarks.dat')
DEFAULT_DLIB_FACE_DETECTOR_PATH = os.path.join(MODEL_DIR, 'dlib', 'mmod_human_face_detector.dat')
DEFAULT_OPENFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'openface', 'nn4.small2.v1.pt')
IMG_DIM = 96


class OpenFaceDataset(Dataset):
    def __init__(self, aligned_dataset_dir, annotations_file=None, transform=None, target_transform=None):
        self.dataset_dir = aligned_dataset_dir
        if annotations_file is None:
            class_folders = [sub.name for sub in os.scandir(aligned_dataset_dir) if sub.is_dir()]
            img_label_list = []
            for class_name in class_folders:
                class_path = os.path.join(aligned_dataset_dir, class_name)
                for img in os.scandir(class_path):
                    if img.name.lower().split('.')[-1] in SUPPORTED_IMAGE_EXTENSIONS:
                        img_label_list.append({'filename': os.path.join(class_path, img.name),
                                         'label': class_name})
            self.img_labels = pd.DataFrame(img_label_list)
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
            raise Exception('Unable to load image: {}'.format(img_path))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            rgb_img = self.transform(rgb_img)
        if self.target_transform:
            label = self.target_transform(label)
        return rgb_img, label


def transform_image(image):
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


def align_all_images(raw_dataset_dir, align_dir, align, landmark_indices, skip_multi=False):
    class_folders = [sub.name for sub in os.scandir(raw_dataset_dir) if sub.is_dir()]
    print('=== Detecting and aligning faces ===')
    summary_str = '{:<16}{:>8}\n'.format('Name', 'Count')
    summary_str += '-' * 24
    for class_name in class_folders:
        raw_class_path = os.path.join(raw_dataset_dir, class_name)
        aligned_class_path = os.path.join(align_dir, class_name)
        os.makedirs(aligned_class_path, exist_ok=True)
        aligned_count = 0
        for img in os.scandir(raw_class_path):
            if img.name.lower().split('.')[-1] in SUPPORTED_IMAGE_EXTENSIONS:
                img_path = os.path.join(raw_class_path, img.name)
                bgr_img = cv2.imread(img_path)
                if bgr_img is None:
                    print('Warning: Unable to load image: {}'.format(img_path))
                    continue
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                aligned_rgb_img = align.align(IMG_DIM, rgb_img, landmarkIndices=landmark_indices,
                                              skipMulti=skip_multi)
                if aligned_rgb_img is None:
                    print('Warning: Unable to find a face: {}'.format(img_path))
                    continue
                aligned_bgr_img = cv2.cvtColor(aligned_rgb_img, cv2.COLOR_RGB2BGR)
                aligned_img_path = os.path.join(aligned_class_path, img.name)
                cv2.imwrite(aligned_img_path, aligned_bgr_img)
                aligned_count += 1
        summary_str += '\n{:<16}{:>8}'.format(class_name, aligned_count)
    print(summary_str)


def main(args):
    input_dataset_dir = args.input_dir
    output_csv_dir = args.csv_out
    reps_csv_path = os.path.join(output_csv_dir, REPS_CSV_FILE)
    labels_csv_path = os.path.join(output_csv_dir, LABELS_CSV_FILE)
    os.makedirs(output_csv_dir, exist_ok=True)
    for csv_path in [reps_csv_path, labels_csv_path]:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    if args.aligned:
        dataset = OpenFaceDataset(input_dataset_dir, transform=transform_image)
    else:
        output_align_dir = args.align_out
        os.makedirs(output_align_dir, exist_ok=True)
        if args.dlib_face_detector_type == 'CNN':
            align = openface.AlignDlib(args.dlib_face_predictor_path, args.dlib_face_detector_path,
                                       upsample=args.upsample)
        else:
            align = openface.AlignDlib(args.dlib_face_predictor_path, upsample=args.upsample)
        landmark_map = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if args.landmarks not in landmark_map:
            raise Exception('Landmarks unrecognized: {}'.format(args.landmarks))
        landmark_indices = landmark_map[args.landmarks]

        align_all_images(input_dataset_dir, output_align_dir, align, landmark_indices, args.skip_multi)
        dataset = OpenFaceDataset(output_align_dir, transform=transform_image)

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=args.shuffle, num_workers=args.worker)
    model = openface.OpenFaceNet()
    if args.cpu:
        model.load_state_dict(torch.load(args.openface_model_path))
    else:
        model.load_state_dict(torch.load(args.openface_model_path, map_location='cuda'))
        model.to(torch.device('cuda'))
    model.eval()

    label_dict = {}
    label_counter = Counter()
    for step, (images, labels) in enumerate(dataloader):
        print('=== Generating representations for batch {}/{} ==='.format(step, len(dataloader)))
        if not args.cpu:
            images = images.to(torch.device('cuda'))
        reps = model(images)
        reps = reps.cpu().detach().numpy()

        with open(reps_csv_path, 'a') as reps_file:
            np.savetxt(reps_file, reps, fmt='%.8f', delimiter=',')

        label_counter.update(labels)
        with open(labels_csv_path, 'a') as labels_file:
            for label in labels:
                labels_file.write('{},{}\n'.format(get_or_add(label, label_dict), label))
    print('Summary: Representations generated for {} images in total'.format(sum(label_counter.values())))
    print(dict(label_counter))
    print('Saving csv files to folder: "{}"'.format(output_csv_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to image dataset directory')
    parser.add_argument('-o', '--csv_out', required=True, type=str, help='path to csv output directory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--aligned', action='store_true',
                        help='if flag is set, assume input images are already aligned')
    group.add_argument('--align_out', type=str, help='save aligned images to the specified directory')
    parser.add_argument('--dlib_face_predictor_path', type=str, default=DEFAULT_DLIB_FACE_PREDICTOR_PATH,
                        help='path to dlib face predictor model')
    parser.add_argument('--dlib_face_detector_type', type=str, choices=['HOG', 'CNN'], default='CNN',
                        help='type of dlib face detector to be used')
    parser.add_argument('--dlib_face_detector_path', type=str, default=DEFAULT_DLIB_FACE_DETECTOR_PATH,
                        help='path to dlib CNN face detector model')
    parser.add_argument('--upsample', type=int,  default=1, help="number of times to upsample images before detection.")
    parser.add_argument('--openface_model_path', type=str, default=DEFAULT_OPENFACE_MODEL_PATH,
                        help='path to pretrained OpenFace model')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
    parser.add_argument('--skip_multi', action='store_true', help='if flag is set, skip image if multiple faces are'
                                                                  'found, otherwise only use the largest face')
    parser.add_argument('--landmarks', type=str, choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                        default='outerEyesAndNose', help='landmarks to align to')
    parser.add_argument('--cpu', action='store_true', help='run OpenFace model on CPU only')
    arguments = parser.parse_args()

    main(arguments)
