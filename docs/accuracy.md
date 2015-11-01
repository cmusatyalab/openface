# Cool demos, but I want numbers. What's the accuracy?
Even though the public datasets we trained on have orders of magnitude less data
than private industry datasets, the accuracy is remarkably high
on the standard
[LFW](http://vis-www.cs.umass.edu/lfw/results.html)
benchmark.
We had to fallback to using the deep funneled versions for
152 of 13233 images because dlib failed to detect a face or landmarks.
We obtain a mean accuracy of 0.8483 &plusmn; 0.0172 with an AUC of 0.923.
For comparison, training with Google-scale data results in an
accuracy of .9963 &plusmn; 0.009.

![](../../images/nn4.v1.lfw.roc.png)

This can be generated with the following commands from the root `openface`
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
3. Generate representations with `./batch-represent/main.lua -outDir evaluation/lfw.nn4.v1.reps -model models/openface/nn4.v1.t7 -data data/lfw/dlib-affine-sz:96`
4. Generate the ROC curve from the `evaluation` directory with `./lfw-roc.py --workDir lfw.nn4.v1.reps`.
   This creates `roc.pdf` in the `lfw.nn4.v1.reps` directory.

---

If you're interested in higher accuracy open source code, see:

1. [Oxford's VGG Face Descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/),
  which is licensed for non-commercial research purposes.
  They've released their softmax network, which obtains .9727 accuracy
  on the LFW and will release their triplet network (0.9913 accuracy)
  and data soon.

  Their softmax model doesn't embed features like FaceNet,
  which makes tasks like classification and clustering more difficult.
  Their triplet model hasn't yet been released, but will provide
  embeddings similar to FaceNet.
  The triplet model will be supported by OpenFace once it's released.
2. [AlfredXiangWu/face_verification_experiment](https://github.com/AlfredXiangWu/face_verification_experiment),
  which uses Caffe and doesn't yet have a license.
  The accuracy on the LFW is .9777.
  This model doesn't embed features like FaceNet,
  which makes tasks like classification and clustering more difficult.
