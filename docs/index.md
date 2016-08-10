# OpenFace <iframe src="https://ghbtns.com/github-btn.html?user=cmusatyalab&repo=openface&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>

<center>
*Free and open source face recognition with
deep neural networks.*
</center>

---

## News

+ 2016-08-09: [New blog post: (Face) Image Completion with Deep Learning in TensorFlow](https://groups.google.com/forum/#!topic/cmu-openface/h7t-URw7zJA)
+ 2016-06-01: [OpenFace tech report released](http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf)
+ 2016-01-19: OpenFace 0.2.0 released!
  See [this blog post](http://bamos.github.io/2016/01/19/openface-0.2.0/)
  for more details.

---

OpenFace is a Python and [Torch](http://torch.ch) implementation of
face recognition with deep neural networks and is based on
the CVPR 2015 paper
[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google.
Torch allows the network to be executed on a CPU or with CUDA.

**Crafted by [Brandon Amos](http://bamos.github.io),
[Bartosz Ludwiczuk](https://github.com/melgor), and
[Mahadev Satyanarayanan](https://www.cs.cmu.edu/~satya/).**

---

+ The code is available on GitHub at
  [cmusatyalab/openface](https://github.com/cmusatyalab/openface).
+ [API Documentation](http://openface-api.readthedocs.org/en/latest/index.html)
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
See [our accuracy comparisons](http://cmusatyalab.github.io/openface/models-and-accuracies/)
on the famous LFW benchmark.

---

### Please use responsibly!

We do not support the use of this project in applications
that violate privacy and security.
We are using this to help cognitively impaired users
sense and understand the world around them.

---

# Overview

The following overview shows the workflow for a single input
image of Sylvestor Stallone from the publicly available
[LFW dataset](http://vis-www.cs.umass.edu/lfw/person/Sylvester_Stallone.html).

1. Detect faces with a pre-trained models from
  [dlib](http://blog.dlib.net/2014/02/dlib-186-released-make-your-own-object.html)
  or
  [OpenCV](http://docs.opencv.org/master/tutorial_py_face_detection.html).
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


# Posts About OpenFace

+ [July 24, 2016] [Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.ds8i8oic9)
+ [Feb 24, 2016] [Hey Zuck, We Built Your Office A.I. Solution](http://blog.algorithmia.com/2016/02/hey-zuck-we-built-your-facial-recognition-ai/)
+ [Feb 3, 2016] [RTNiFiOpenFace and WebSocketServer add face recognition to an Apache NiFi video flow](https://richardstechnotes.wordpress.com/2016/02/03/rtnifiopenface-and-websocketserver-add-face-recognition-to-an-apache-nifi-video-flow/)
+ [Jan 29, 2016] [Integrating OpenFace into an Apache NiFi flow using WebSockets](https://richardstechnotes.wordpress.com/2016/01/29/integrating-openface-into-an-apache-nifi-flow-using-websockets/)
+ [Oct 15, 2015] (Spanish) GenBeta: [OpenFace, un nuevo software de reconocimiento facial, de código abierto](http://www.genbeta.com/actualidad/openface-un-nuevo-software-de-reconocimiento-facial-de-codigo-abierto)
+ [Oct 15, 2015] TheNextWeb: [Watch this open-source program recognize faces in real time](http://thenextweb.com/dd/2015/10/15/watch-this-open-source-program-recognize-faces-in-real-time/)

# Notable Relevant Projects
+ [davidsandberg/facenet](https://github.com/davidsandberg/facenet):
  FaceNet TensorFlow implementation.
+ [pyannote/pyannote-video](https://github.com/pyannote/pyannote-video):
  Face detection, tracking, and clustering in videos using OpenFace.
+ [aybassiouny/OpenFaceCpp](https://github.com/aybassiouny/OpenFaceCpp):
  Unofficial OpenFace C++ implementation and bindings.

# Citations

Please cite OpenFace in your publications if it helps your research.
The following is a [BibTeX](http://www.bibtex.org/) and plaintext reference for our
[OpenFace tech report](http://reports-archive.adm.cs.cmu.edu/anon/anon/2016/CMU-CS-16-118.pdf).

```
@techreport{amos2016openface,
  title={OpenFace: A general-purpose face recognition
    library with mobile applications},
  author={Amos, Brandon and Bartosz Ludwiczuk and Satyanarayanan, Mahadev},
  year={2016},
  institution={CMU-CS-16-118, CMU School of Computer Science},
}

B. Amos, B. Ludwiczuk, M. Satyanarayanan,
"Openface: A general-purpose face recognition library with mobile applications,"
CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.
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
+ The GitHub issue and pull request templates are inspired from
  [Randy Olsen's](http://www.randalolson.com/) templates at [rhiever/tpot](https://github.com/rhiever/tpot),
  [Justin Abrahms'](https://justin.abrah.ms/) [PR template](https://quickleft.com/blog/pull-request-templates-make-code-review-easier/),
  and
  [Aurelia Moser's](http://algorhyth.ms/) [issue template](https://bl.ocks.org/auremoser/72803ba969d0e61ff070).
+ Zhuo Chen, Kiryong Ha, Wenlu Hu,
  [Rahul Sukthankar](http://www.cs.cmu.edu/~rahuls/), and
  Junjue Wang for insightful discussions.

# Licensing
Unless otherwise stated, the source code and trained Torch and Python
model files are copyright Carnegie Mellon University and licensed
under the
[Apache 2.0 License](https://github.com/cmusatyalab/openface/blob/master/LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | Modified | License
---|---|---|
[Atcold/torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding) | No | MIT
[facebook/fbnn](https://github.com/facebook/fbnn) | Yes | BSD
