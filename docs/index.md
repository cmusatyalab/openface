# OpenFace <iframe src="https://ghbtns.com/github-btn.html?user=cmusatyalab&repo=openface&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>

<center>
*Free and open source face recognition with
deep neural networks.*
</center>

---

OpenFace is a Python and [Torch](http://torch.ch) implementation of
face recognition with deep neural networks and is based on
the CVPR 2015 paper
[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google.
Torch allows the network to be executed on a CPU or with CUDA.

**Crafted by [Brandon Amos](http://bamos.github.io) in the
[Elijah](http://elijah.cs.cmu.edu) research group at
Carnegie Mellon University.**

---

+ The code is available on GitHub at
  [cmusatyalab/openface](https://github.com/cmusatyalab/openface).
+ [API Documentation](http://openface-api.readthedocs.org/en/latest/openface.html)
+ Join the
  [cmu-openface group](https://groups.google.com/forum/#!forum/cmu-openface)
  or the
  [gitter chat](https://gitter.im/cmusatyalab/openface)
  for discussions and installation issues.
+ Development discussions and bugs reports are on the
  [issue tracker](https://github.com/cmusatyalab/openface/issues).

---

This research was supported by the National Science Foundation (NSF)
under grant number CNS-1518865.  Additional support
was provided by the Intel Corporation, Google, Vodafone, NVIDIA, and the
Conklin Kistler family fund.  Any opinions, findings, conclusions or
recommendations expressed in this material are those of the authors
and should not be attributed to their employers or funding sources.

---

### Isn't face recognition a solved problem?
No! Accuracies from research papers have just begun to surpass
human accuracies on some benchmarks.
The accuracies of open source face recognition systems lag
behind the state-of-the-art.
See [our accuracy comparisons](http://cmusatyalab.github.io/openface/accuracy/)
on the famous LFW benchmark.

---

### Please use responsibly!

We do not support the use of this project in applications
that violate privacy and security.
We are using this to help cognitively impaired users to
sense and understand the world around them.

---

# Overview

The following overview shows the workflow for a single input
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
   to try to make the eyes and bottom lip appear in
   the same location on each image.
3. Use a deep neural network to represent (or embed) the face on
   a 128-dimensional unit hypersphere.
   The embedding is a generic representation for anybody's face.
   Unlike other face representations, this embedding has the nice property
   that a larger distance between two face embeddings means
   that the faces are likely not of the same person.
   This property makes clustering, similarity detection,
   and classification tasks easier than other face recognition
   techniques where the Euclidean distance between
   features is not meaningful.
4. Apply your favorite clustering or classification techniques
   to the features to complete your recognition task.
   See below for our examples for classification and
   similarity detection, including an online web demo.

![](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/summary.jpg)

# Citations

The following is a [BibTeX](http://www.bibtex.org/)
and plaintext reference
for the OpenFace GitHub repository.
The reference may change in the future.
The BibTeX entry requires the `url` LaTeX package.

```
@misc{amos2015openface,
    title        = {{OpenFace: Face Recognition with Deep Neural Networks}},
    author       = {Amos, Brandon and Harkes, Jan and Pillai, Padmanabhan and Elgazzar, Khalid and Satyanarayanan, Mahadev},
    howpublished = {\url{http://github.com/cmusatyalab/openface}},
    note         = {Accessed: 2015-11-11}
}

Brandon Amos, Jan Harkes, Padmanabhan Pillai, Khalid Elgazzar,
and Mahadev Satyanarayanan.
OpenFace: Face Recognition with Deep Neural Networks.
http://github.com/cmusatyalab/openface.
Accessed: 2015-11-11.
```

# Acknowledgements
+ The fantastic Torch ecosystem and community.
+ [Alfredo Canziani's](https://github.com/Atcold)
  implementation of FaceNet's loss function in
  [torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding).
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
+ [Davis King's](https://github.com/davisking) [dlib](https://github.com/davisking/dlib)
  library for face detection and alignment.
+ Zhuo Chen, Kiryong Ha, Wenlu Hu,
  [Rahul Sukthankar](http://www.cs.cmu.edu/~rahuls/), and
  Junjue Wang for insightful discussions.

# Licensing
The source code and trained models `nn4.v1.t7` and
`celeb-classifier.nn4.v1.t7` are copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](https://github.com/cmusatyalab/openface/tree/master/LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | Modified | License
---|---|---|
[Atcold/torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding) | No | MIT
[facebook/fbnn](https://github.com/facebook/fbnn) | Yes | BSD
