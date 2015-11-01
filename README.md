# OpenFace

[ ![Build Status] [travis-image] ] [travis]
[ ![Release] [release-image] ] [releases]
[ ![License] [license-image] ] [license]
[ ![DOI] [doi-image] ] [doi]
[ ![Gitter] [gitter-image] ] [gitter]
[ ![Documentation Status][doc-image] ] [docs]

[travis-image]: https://travis-ci.org/cmusatyalab/openface.png?branch=master
[travis]: http://travis-ci.org/cmusatyalab/openface

[release-image]: http://img.shields.io/badge/release-0.1.1-blue.svg?style=flat
[releases]: https://github.com/cmusatyalab/openface/releases

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

[doi-image]: https://zenodo.org/badge/doi/10.5281/zenodo.32148.svg
[doi]: http://dx.doi.org/10.5281/zenodo.32148

[gitter-image]: https://badges.gitter.im/Join%20Chat.svg
[gitter]: https://gitter.im/cmusatyalab/openface

[doc-image]: https://readthedocs.org/projects/openface/badge/?version=latest
[docs]: http://openface.readthedocs.org/en/latest/docs?badge=latest


This is a Python and [Torch](http://torch.ch) implementation of the CVPR 2015 paper
[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google
using publicly available libraries and datasets.
Torch allows the network to be executed on a CPU or with CUDA.

**Crafted by [Brandon Amos](http://bamos.github.io) in the
[Elijah](http://elijah.cs.cmu.edu) research group at
Carnegie Mellon University.**

---

## Overview

![](./images/summary.jpg)

---

## Real-Time Demo

<a href='https://www.youtube.com/watch?v=LZJOTRkjZA4'><img src='images/youtube-web.gif'></img></a>

---

The documentation is available [here](http://openface.readthedocs.org/en/latest/docs).

Please join the
[cmu-openface group](https://groups.google.com/forum/#!forum/cmu-openface)
or the
[gitter chat](https://gitter.im/cmusatyalab/openface)
for discussions and installation issues.

Development discussions and bugs reports are on the
[issue tracker](https://github.com/cmusatyalab/openface/issues).

# Citations

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.32041.svg)](http://dx.doi.org/10.5281/zenodo.32041)

Please cite this repository if you use this in academic works.


```
@misc{amos2015openface,
    author       = {Amos, Brandon and Harkes, Jan and Pillai, Padmanabhan and Elgazzar, Khalid and Satyanarayanan, Mahadev},
    title        = {OpenFace 0.1.1: Face recognition with Google's FaceNet deep neural network},
    month        = oct,
    year         = 2015,
    doi          = {10.5281/zenodo.32148},
    url          = {http://dx.doi.org/10.5281/zenodo.32148}
}
```

# Acknowledgements
+ The fantastic Torch ecosystem and community.
+ [Alfredo Canziani's](https://github.com/Atcold)
  implementation of FaceNet's loss function in
  [torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding)
+ [Nicholas Léonard](https://github.com/nicholas-leonard)
  for quickly merging my pull requests to
  [nicholas-leonard/dpnn](https://github.com/nicholas-leonard/dpnn)
  modifying the inception layer.
+ [Francisco Massa](https://github.com/fmassa)
  and
  [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)
  for
  quickly releasing [nn.Normalize](https://github.com/torch/nn/pull/341)
  after I expressed interest in using it.
+ [Soumith Chintala](https://github.com/soumith) for
  help with the [fbcunn](https://github.com/facebook/fbcunn)
  example code.
+ NVIDIA's academic
  [hardware grant program](https://developer.nvidia.com/academic_hw_seeding)
  for providing the Tesla K40 used to train the model.
+ [Davis King's](https://github.com/davisking) [dlib](https://github.com/davisking/dlib)
  library for face detection and alignment.
+ Zhuo Chen, Kiryong Ha, Wenlu Hu,
  [Rahul Sukthankar](http://www.cs.cmu.edu/~rahuls/), and
  Junjue Wang for insightful discussions.

# Licensing
The source code and trained models `nn4.v1.t7` and
`celeb-classifier.nn4.v1.t7` are copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | Modified | License
---|---|---|
[Atcold/torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding) | No | MIT
[facebook/fbnn](https://github.com/facebook/fbnn) | Yes | BSD
