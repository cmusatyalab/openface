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

One option could be to have all of your data in `train` and
then validate the model with the LFW experiment.

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
Install the Python dependencies from
[training/requirements.txt](https://github.com/cmusatyalab/openface/blob/master/training/requirements.txt)
with `pip2 install -r requirements.txt`.


# Discrepancies between OpenFace and FaceNet training

From [Bartosz Ludwiczuk](https://github.com/melgor) in
[this](https://groups.google.com/d/msg/cmu-openface/dcPh883T1rk/5m53axGzAwAJ)
mailing list post.

1. "we use all anchor-positive pairs": current in pipeline is use only one
possible combination between anchor-positive (each anchor get one random
positive example). Here should be created all possible pairs, so much more
than now.

2. "In order to ensure fast convergence it is crucial to select triplets
that violate the triplet constraint in Eq.1": so for evaluation only
triplets which violate constraint should be used. Why? Because other
triplets produce "0" gradient, which then lower the final gradient update
in model (so, all chosen triplet should be in margin, other should not be
considered). I think which I am not sure is merging the gradient from
triplets to the model. I have doubts if the gradient from each sample
should be averaged (as one sample could be used several time) or just live
it as is.

3. Based on both points, current pipeline of processing image should be
  changed (I mean using three copy of models is wrong idea). As we want to
  generate all possible triplets, the pipeline should be following:

    - one model forward all images in one pass - maximum number based on
    memory consumption, for 4 GB it would be 140 for nn4, for 12 GB it could be
    ~ 450
    - create all possible positive pairs using embeding (remember truth idx
    from embeding matrix)- for batch size 10 people each 14 images gives ~800
    pairs
    - choose random negative example which violate the triplet constraint.
    If pair have no such negative example, remove it
    - go through criteria and calculate gradient (forward and backward pass)
    - map the gradient from triplets to embedding
    - make a backward pass through model

    It is much faster than current version and it can be run at single GPU

4. As CASIA-WebFace is much smaller than Google data, I think that choosing
the the triplets only in margin will slow down the convergence. It is much
better to use idea from Oxford-Face and choose random triplet which violate
constraint (so negative example can be closer to anchor than positive).

5. The testing procedure is hard, I think there should not be sth like
"random triplets", such number does not inform us about performance of
model. I was thinking about implementing checking the LFW score after each
epoch (as we use such metric to evaluate model). This will clearly show if
model go into right direction (Google use Verification accuracy too, but
they use their data).

6. I more note from paper, which I do not implement: "Additionally,
randomly sampled negative faces are added to each mini-batch". It could
boost performance, but not sure if should be implemented now.
