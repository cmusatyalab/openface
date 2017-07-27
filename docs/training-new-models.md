# Training new neural network models

We have also released our deep neural network (DNN)
training infrastructure to promote an open ecosystem and enable quicker
bootstrapping for new research and development.

There is a distinction between training the DNN model for feature representation
and training a model for classifying people with the DNN model.
If you're interested in creating a new classifier,
see [Demo 3](http://cmusatyalab.github.io/openface/demo-3-classifier/).
This page is for advanced users interested in training a new DNN model
and should be done with large datasets (>500k images) to improve the
feature representation.

*Warning:* Training is computationally and memory expensive and takes a
day on our Tesla K40 GPU.

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
If you plan to compute LFW accuracies, remove all LFW identities for your dataset.
We provide an example script doing this with string matching in
[remove-lfw-names.py](https://github.com/cmusatyalab/openface/blob/master/data/casia-facescrub/remove-lfw-names.py).

Change `8` to however many
separate processes you want to run:
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align outerEyesAndNose <path-to-aligned-data> --size 96 & done`.

Prune out directories with less than 3 images per class with
`./util/prune-dataset.py <path-to-aligned-data> --numImagesThreshold 3`.

<!-- Split the dataset into `train` and `val` subdirectories -->
<!-- with `./util/create-train-val-split.py <path-to-aligned-data> <validation-ratio>`. -->
<!-- One option could be to have all of your data in `train` and -->
<!-- then validate the model with the LFW experiment. -->

## 3. Train the model
Run [training/main.lua](https://github.com/cmusatyalab/openface/blob/master/training/main.lua) to start training the model.
Edit the dataset options in [training/opts.lua](https://github.com/cmusatyalab/openface/blob/master/training/opts.lua) or
pass them as command-line parameters.
This will output the loss and in-progress models to `training/work`.
The GPU memory usage is determined by the `-peoplePerBatch` and
`-imagesPerPerson` parameters, which default to 15 and 20 respectively
and consume about 12GB of memory.
These determine an upper-bound on the mini-batch size and
should be reduced for less GPU memory consumption.

Warning: Metadata about the on-disk data is cached in
`training/work/trainCache.t7` and assumes
the data directory does not change.
If your data directory changes, delete these
files so they will be regenerated.

### Stopping and starting training
Models are saved in the `work` directory after every epoch.
If the training process is killed, it can be resumed from
the last saved model with the `-retrain` option.
Also pass a different `-manualSeed` so a different image
sequence is sampled and correctly set `-epochNumber`.

## 4. Analyze training
Visualize the loss with [training/plot-loss.py](https://github.com/cmusatyalab/openface/blob/master/training/plot-loss.py).
Install the Python dependencies from
[training/requirements.txt](https://github.com/cmusatyalab/openface/blob/master/training/requirements.txt)
with `pip2 install -r requirements.txt`.
