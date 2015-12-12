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
few weeks on our Tesla K40 GPU.
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
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align innerEyesAndBottomLip <path-to-aligned-data> --size 96 & done`.
Prune out directories with less than N (I use 10) images
per class with `./util/prune-dataset.py <path-to-aligned-data> --numImagesThreshold <N>` and
then split the dataset into `train` and `val` subdirectories
with `./util/create-train-val-split.py <path-to-aligned-data> <validation-ratio>`.

## 3. Train the model
Run [training/main.lua](https://github.com/cmusatyalab/openface/blob/master/training/main.lua) to start training the model.
Edit the dataset options in [training/opts.lua](https://github.com/cmusatyalab/openface/blob/master/training/opts.lua) or
pass them as command-line parameters.
This will output the loss and in-progress models to `training/work`.
The default minibatch size (parameter `-batchSize`) is 100 and requires
about 10GB of GPU memory.

Warning: Metadata about the on-disk data is cached in
`training/work/{train,test}Cache.t7` and assumes
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
