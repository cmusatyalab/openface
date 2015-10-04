# FaceNet

This is a Python and Torch implementation of the CVPR 2015 paper
[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google
using publicly available libraries and datasets.
Torch allows the network to be executed on a CPU or with CUDA.

**Crafted by [Brandon Amos](http://bamos.github.io) in the
[Elijah](http://elijah.cs.cmu.edu) research group at
Carnegie Mellon University.**

---

### Isn't face recognition a solved problem?
No! Accuracies from research papers have just begun to surpass
human accuracies on some benchmarks.
The accuracies of open source face recognition systems lag
behind the state-of-the-art.
See our accuracy comparisons on the famous LFW benchmark below.

---

The following example shows the workflow for a single input
image of Sylvestor Stallone from the publicly available
[LFW dataset](http://vis-www.cs.umass.edu/lfw/person/Sylvester_Stallone.html).

1. Detect faces with a pre-trained models from
  [dlib](http://blog.dlib.net/2014/02/dlib-186-released-make-your-own-object.html)
  or
  [OpenCV](http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html).
2. Transform the face for the neural network.
   This repository uses dlib's
   [real-time pose estimation](http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html)
   with OpenCV's
   [affine transformation](http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html)
   to try to make the eyes and nose appear in
   the same location on each image.
3. Use a deep neural network to represent (or embed) the face on
   a 128-dimensional unit hypersphere.
   The embedding is a generic representation for anybody's face.
   Unlike other face representations, this embedding has the nice property
   that a larger distance between two face embeddings means
   that the faces are likely not of the same person.
   This trivializes clustering, similarity detection,
   and classification tasks.

![](./images/summary.jpg)

# Help Wanted!

As the following table shows, the forefront of deep learning research
is driven by large private datasets.
In face recognition, there are no open source implementations or
models trained on these datasets.
If you have access to a large dataset, we are very interested
in training a new FaceNet model with it.
Please contact Brandon Amos at [bamos@cs.cmu.edu](mailto:bamos@cs.cmu.edu).

| Dataset | Public | #Photos | #People |
|---|---|---|---|
| [DeepFace](https://research.facebook.com/publications/480567225376225/deepface-closing-the-gap-to-human-level-performance-in-face-verification/) (Facebook) | No | 4.4 Million | 4k |
| [Web-Scale Training...](http://arxiv.org/abs/1406.5266) (Facebook) | No | 500 Million | 10 Million |
| FaceNet (Google) | No | 100-200 Million | 8 Million |
| [FaceScrub](http://vintage.winklerbros.net/facescrub.html) | Yes | 100k | 500 |
| [CASIA-WebFace](http://arxiv.org/abs/1411.7923) | Yes | 500k | 10k |

# What's in this repository?
+ [batch-represent](/batch-represent): Generate representations from
  a batch of images, stored in a directory by names.
+ [demos/www](/demos/www): Real-time web demo.
+ [demos/compare.py](/demos/compare.py): Compare two images.
+ [evaluation](/evaluation): LFW accuracy evaluation scripts.
+ [facenet](/facenet): Python library code.
+ [images](/images): Images used in the README.
+ [models](/models): Location of binary models.
+ [training](/training): Scripts to train new models.
+ [util](/util): Utility scripts.

# Real-Time Web Demo
See [our YouTube video](https://www.youtube.com/watch?v=uQiPq5zRaS8)
of using this in a real-time web application
for face recognition.
The source is available in [demos/web](/demos/web).

<a href='https://www.youtube.com/watch?v=uQiPq5zRaS8'><img src='images/youtube-web.gif'></img></a>

From the `demos/web` directory, install requirements
with `./install-deps.sh` and `sudo pip install -r requirements.txt`.

# Comparing two images
The [comparison demo](demos/compare.py) outputs the predicted similarity
score of two faces by computing the squared L2 distance between
their representations.
A lower score indicates two faces are more likely of the same person.
Since the representations are on the unit hypersphere, the
scores range from 0 (the same picture) to 4.0.
The following distances between images of John Lennon and
Eric Clapton were generated with
`./demos/compare.py images/examples/{lennon*,clapton*}`.

| Lennon 1 | Lennon 2 | Clapton 1 | Clapton 2 |
|---|---|---|---|
| <img src='images/examples/lennon-1.jpg' width='200px'></img> | <img src='images/examples/lennon-2.jpg' width='200px'></img> | <img src='images/examples/clapton-1.jpg' width='200px'></img> | <img src='images/examples/clapton-2.jpg' width='200px'></img> |

The following table shows that a distance threshold of `0.3` would
distinguish these two images.
In practice, further experimentation should be done on the distance threshold.
On our LFW experiments, a threshold of `0.70` (TODO: Update when used on the final model)
gave the best accuracy on 8 out of 10 experiments.

| Image 1 | Image 2 | Distance |
|---|---|---|
| Lennon 1 | Lennon 2 | 0.204 |
| Lennon 1 | Clapton 1 | 1.392 |
| Lennon 1 | Clapton 2 | 1.445 |
| Lennon 2 | Clapton 1 | 1.435 |
| Lennon 2 | Clapton 2 | 1.322 |
| Clapton 1 | Clapton 2 | 0.174 |


# Cool demos, but I want numbers. What's the accuracy?
Even though the public datasets we trained on have orders of magnitude less data
than private industry datasets, the accuracy is remarkably high and
outperforms all other open-source face recognition implementations we
are aware of on the standard
[LFW](http://vis-www.cs.umass.edu/lfw/results.html)
benchmark.
We had to fallback to using the deep funneled versions for
152 of 13233 images because dlib failed to detect a face or landmarks.

[TODO: Add final ROC Curve]

This can be generated with the following commands from the root `facenet`
directory, assuming you have downloaded and placed the raw and
deep funneled LFW data from [here](http://vis-www.cs.umass.edu/lfw/)
in `./data/lfw/raw` and `./data/lfw/deepfunneled`.

1. Install prerequisites as below.
2. Preprocess the raw `lfw` images, change `8` to however many
   separate processes you want to run:
   `for N in {1..8}; do ./util/align-dlib.py data/lfw/raw align affine data/lfw/dlib-affine-sz:96 --size 96 &; done`.
   Fallback to deep funneled versions for images that dlib failed
   to align:
   `./util/align-dlib.py data/lfw/raw align affine data/lfw/dlib-affine-sz:96 --size 96 --fallbackLfw data/lfw/deepfunneled`
3. Generate representations with `./batch-represent/main.lua -outDir evaluation/lfw.nn4.v1.reps -model models/facenet/nn4.v1.t7 -data data/lfw/dlib-affine-sz:96`
4. Generate the ROC curve from the `evaluation` directory with `./lfw-roc.py --workDir lfw.nn4.v1.reps`.
   This creates `roc.pdf` in the `lfw.nn4.v1.reps` directory.

# Visualizing representations t-SNE
[t-SNE](http://lvdmaaten.github.io/tsne/) is a dimensionality
reduction technique that can be used to visualize the
128-dimensional features FaceNet produces.
The following shows the visualization of the three people
in the training and testing dataset with the most images.

**Training**

![](images/train-tsne.png)

**Testing**

![](images/val-tsne.png)

These can be generated with the following commands from the root
`facenet` directory.

1. Install prerequisites as below.
2. Preprocess the raw `lfw` images, change `8` to however many
   separate processes you want to run:
   `for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align affine <path-to-aligned-data> --size 96 &; done`.
3. Generate representations with `./batch-represent/main.lua -outDir <feature-directory (to be created)> -model models/facenet/nn4.v1.t7 -data <path-to-aligned-data>`
4. Generate t-SNE visualization with `./util/tsne.py <feature-directory> --names <name 1> ... <name n>`
   This creates `tsne.pdf` in `<feature-directory>`.

# Model Definitions
Model definitions should be kept in [models/facenet](models/facenet),
where we have provided definitions of the [nn1](models/facenet/nn1.def.lua)
and [nn4](models/facenet/nn4.def.lua) as described in the paper,
but with batch normalization and no normalization in the lower layers.

# Pre-trained Models
Pre-trained models are versioned and should be released with
a corresponding model definition.
We currently only provide a pre-trained model for `nn4.v1`
because we have limited access to large-scale face recognition
datasets.

## nn4.v1
This model has been trained by combining the two largest (of August 2015)
publicly-available face recognition datasets based on names:
[FaceScrub](http://vintage.winklerbros.net/facescrub.html)
and [CASIA-WebFace](http://arxiv.org/abs/1411.7923).
This model was trained for TODO hours on a Tesla K40 GPU.

The following plot shows the triplet loss on the training
and test set.
Semi-hard triplets are used on the training set, and
random triplets are used on the testing set.

[TODO: Loss plot]

The LFW section above shows that this model obtains a mean
accuracy of TODO with an AUC of TODO.

# Setup
The following instructions are for Linux and OSX only.
Please contribute modifications and build instructions if you
are interested in running this on other operating systems.

## Check out git submodules
Clone with `--recursive` or run `git submodule init && git submodule update`
after checking out.

## Download the models
Run `./models/get-models.sh` to download pre-trained FaceNet
models on the combined CASIA-WebFace and FaceScrub database.
This also downloads dlib's pre-trained model for face landmark detection.

## With Docker
Be sure you have checked out the submodules and downloaded the models as
described above.

This repo can be deployed as a container with [Docker](https://www.docker.com/)
for CPU mode:

```
sudo docker build -t facenet .
sudo docker run -t -i -v $PWD:/facenet facenet /bin/bash
cd /facenet
./demos/compare.py images/examples/{lennon*,clapton*}
```

To use, place your images in `facenet` on your host and
access them from the shared Docker directory.

## By hand
Be sure you have checked out the submodules and downloaded the models as
described above.

The main dependencies from a package manager are Torch and Python 2.
Afterwards, manually install the following.

### OpenCV
Download [OpenCV 2.4.11](https://github.com/Itseez/opencv/archive/2.4.11.zip)
and follow their
[build instructions](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html).

### dlib
Download [dlib v18.16](https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2).

```
cd ~/src
tar xf dlib-18.16.tar.bz2
cd dlib-18.16/python_examples
mkdir build
cd build
cmake ../../tools/python
cmake --build . --config Release
cp dlib.so ..
```

### Torch
Install [Torch](http://torch.ch) from the instructions on their website
and install the [dpnn](https://github.com/nicholas-leonard/dpnn)
and [nn](https://github.com/torch/nn) libraries with
`luarocks install dpnn` and `luarocks install nn`.

If you want CUDA support, also install
[cudnn.torch](https://github.com/soumith/cudnn.torch).

# Usage
## Existing Models
See [util/compare.py](the image comparison demo) for a complete example
written in Python using a naive Torch subprocess to process the faces.

```Python
import facenet
from facenet.alignment import NaiveDlib # Depends on dlib.

# `args` are parsed command-line arguments.

align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = facenet.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

# `img` is a numpy matrix containing the RGB pixels of the image.
bb = align.getLargestFaceBoundingBox(img)
alignedFace = align.alignImg("affine", args.imgDim, img, bb)
rep1 = net.forwardImage(alignedFace)

# `rep2` obtained similarly.
d = rep1 - rep2
distance = np.dot(d, d)
```

# Training new models
This repository also contains our training infrastructure to promote an
open ecosystem and enable quicker bootstrapping for new research and development.
Warning: Training is computationally expensive and takes a few
weeks on our Tesla K40 GPU.

A rough overview of training is:

1. Preprocess the raw images, change `8` to however many
   separate processes you want to run:
   `for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align affine <path-to-aligned-data> --size 96 &; done`.
2. Run [training/main.lua](training/main.lua) to start training the model.
   Edit the dataset options in [training/opts.lua](training/opts.lua) or
   pass them as command-line parameters.
   This will output the loss and in-progress models to `training/work`.
3. Visualize the loss with [training/plot-loss.py](training/plot-loss.py).

# Acknowledgements
+ The fantastic Torch ecosystem and community.
+ [Alfredo Canziani's](https://github.com/Atcold)
  implementation of FaceNet's loss function in
  [torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbeddin)
+ [Nicholas Léonard's](https://github.com/nicholas-leonard)
  inception layer implementation at
  [nicholas-leonard/dpnn](https://github.com/nicholas-leonard/dpnn).
+ [Francisco Massa](https://github.com/fmassa) for
  quickly releasing [nn.Normalize](https://github.com/torch/nn/pull/341)
  after I expressed interest in using it.
+ [Soumith Chintala](https://github.com/soumith) for
  help with the [fbcunn](https://github.com/facebook/fbcunn)
  example code.
+ NVIDIA's academic
  [hardware grant program](https://developer.nvidia.com/academic_hw_seeding)
  for providing the Tesla K40 used to train the model.

# Licensing
This source is copyright Carnegie Mellon University
and licensed under the [Apache 2.0 License](./LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | Modified | License
---|---|---|
[Atcold/torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding) | No | MIT
[facebook/fbnn](https://github.com/facebook/fbnn) | Yes | BSD
