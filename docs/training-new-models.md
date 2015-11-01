# Training new models
This repository also contains our training infrastructure to promote an
open ecosystem and enable quicker bootstrapping for new research and development.
Warning: Training is computationally expensive and takes a few
weeks on our Tesla K40 GPU.
Because of this, the training code assumes CUDA is installed.

A rough overview of training is:

## 1. Create raw image directory.
Create a directory for your raw images so that images from different
people are in different subdirectories. The names of the labels or
images do not matter, and each person can have a different amount of images.
The images should be formatted as `jpg` or `png` and have
a lowercase extension.

```
$ tree data/mydataset/raw
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png
```


## 2. Preprocess the raw images
Change `8` to however many
separate processes you want to run:
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align affine <path-to-aligned-data> --size 96 &; done`.
Prune out directories with less than N (I use 10) images
per class with `./util/prune-dataset.py <path-to-aligned-data> --numImagesThreshold <N>` and
then split the dataset into `train` and `val` subdirectories
with `./util/create-train-val-split.py <path-to-aligned-data> <validation-ratio>`.

## 3. Train the model
Run [training/main.lua](https://github.com/cmusatyalab/openface/blob/master/training/main.lua) to start training the model.
Edit the dataset options in [training/opts.lua](https://github.com/cmusatyalab/openface/blob/master/training/opts.lua) or
pass them as command-line parameters.
This will output the loss and in-progress models to `training/work`.

## 4. Analyze training
Visualize the loss with [training/plot-loss.py](https://github.com/cmusatyalab/openface/blob/master/training/plot-loss.py).
