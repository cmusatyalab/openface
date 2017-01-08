# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 01/12/2016 """

from keras.preprocessing.image import ImageDataGenerator

__author__ = 'cenk'

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    'gamo/data/aligned',
    target_size=(64, 64),
    batch_size=32, save_to_dir='tmp/', save_prefix='aug', save_format='png')

for i,y in train_generator:
    pass
