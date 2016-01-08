# Models and Accuracies
Model definitions should be kept in [models/openface](https://github.com/cmusatyalab/openface/blob/master/models/openface),
where we have provided definitions of the [nn2](https://github.com/cmusatyalab/openface/blob/master/models/openface/nn2.def.lua)
and [nn4](https://github.com/cmusatyalab/openface/blob/master/models/openface/nn4.def.lua) as described in the FaceNet paper,
but with batch normalization.
The inception layers are introduced  in
[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).

# Pre-trained Models
Pre-trained models are versioned and should be released with
a corresponding model definition.
Switch between models with caution because the embeddings
not compatible with each other.

We have trained `nn4.v1` and `nn4.v2` by combining the two largest
(of August 2015) publicly-available face recognition datasets based on names:
[FaceScrub](http://vintage.winklerbros.net/facescrub.html)
and [CASIA-WebFace](http://arxiv.org/abs/1411.7923).

API differences between the models are:

| Model  | alignment `landmarkIndices` |
|---|---|
| nn4.v1 | `openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP` |
| nn4.v2 | `openface.AlignDlib.OUTER_EYES_AND_NOSE` |

## Accuracy on the LFW Benchmark

Even though the public datasets we trained on have orders of magnitude less data
than private industry datasets, the accuracy is remarkably high
on the standard
[LFW](http://vis-www.cs.umass.edu/lfw/results.html)
benchmark.
We had to fallback to using the deep funneled versions for
58 of 13233 images because dlib failed to detect a face or landmarks.

| Model | Accuracy | AUC |
|---|---|---|
| nn4.v1 | 0.9153 ± 0.0170 | 0.893 |
| nn4.v2 | 0.8138 ± 0.0149 | 0.965 |
| FaceNet Paper (Reference) | 0.9963 ± 0.009 | not provided |

![](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/nn4.lfw.roc.png)

---

This can be generated with the following commands from the root `openface`
directory, assuming you have downloaded and placed the raw and
[deep funneled](http://vis-www.cs.umass.edu/deep_funnel.html)
LFW data from [here](http://vis-www.cs.umass.edu/lfw/)
in `./data/lfw/raw` and `./data/lfw/deepfunneled`.

1. Install prerequisites as below.
2. Preprocess the raw `lfw` images, change `8` to however many
   separate processes you want to run:
   `for N in {1..8}; do ./util/align-dlib.py data/lfw/raw align innerEyesAndBottomLip data/lfw/dlib-affine-sz:96 --size 96 & done`.
   Fallback to deep funneled versions for images that dlib failed
   to align:
   `./util/align-dlib.py data/lfw/raw align innerEyesAndBottomLip data/lfw/dlib-affine-sz:96 --size 96 --fallbackLfw data/lfw/deepfunneled`
3. Generate representations with `./batch-represent/main.lua -outDir evaluation/lfw.nn4.v2.reps -model models/openface/nn4.v2.t7 -data data/lfw/dlib-affine-sz:96`
4. Generate the ROC curve from the `evaluation` directory with `./lfw-roc.py --workDir lfw.nn4.v2.reps`.
   This creates `roc.pdf` in the `lfw.nn4.v2.reps` directory.

---

# Projects with Higher Accuracy

If you're interested in higher accuracy open source code, see:

## [Oxford's VGG Face Descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

This is licensed for non-commercial research purposes.
They've released their softmax network, which obtains .9727 accuracy
on the LFW and will release their triplet network (0.9913 accuracy)
and data soon (?).

Their softmax model doesn't embed features like FaceNet,
which makes tasks like classification and clustering more difficult.
Their triplet model hasn't yet been released, but will provide
embeddings similar to FaceNet.
The triplet model will be supported by OpenFace once it's released.

## [AlfredXiangWu/face_verification_experiment](https://github.com/AlfredXiangWu/face_verification_experiment)

This uses Caffe and doesn't yet have a license.
The accuracy on the LFW is .9777.
This model doesn't embed features like FaceNet,
which makes tasks like classification and clustering more difficult.
