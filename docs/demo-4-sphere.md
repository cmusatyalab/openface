## Demo 4: Real-Time Face Embedding Visualization
Released by [Brandon Amos](http://bamos.github.io) and
[Gabriel Farina](https://github.com/gabrfarina) on 2016-09-15.

---

We had a great opportunity
(*thanks to Jan Harkes, Alison Langmead, and Aaron Henderson*)
to present a short OpenFace demo
in the [Data (after)Lives art exhibit](https://uag.pitt.edu/Detail/occurrences/370)
at the University of Pittsburgh,
which investigates the relationship between the human notions of self and
technical alternative, externalized, and malleable representations of identity.
The following video is just a quick example, and a real-time version
is being shown live from Sept 8, 2016 to Oct 14, 2016.
We have released the source code behind this demo in our main
GitHub repository in
[demos/sphere.py](https://github.com/cmusatyalab/openface/blob/master/demos/sphere.py).
This exhibit also features [two other art pieces](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/sphere-demo/exhibits-nosenzo.png)
by [Sam Nosenzo](http://www.pitt.edu/~san76/),
[Alison Langmead](http://www.haa.pitt.edu/person/alison-langmead/),
and [Aaron Henderson](http://www.aaronhenderson.com/) that use OpenFace.


![](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/sphere-demo/demo.gif)

<center>
![](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/sphere-demo/exhibit-amos.png)
</center>

### How this is implemented

This is a short description of our implementation in
[demos/sphere.py](https://github.com/cmusatyalab/openface/blob/master/demos/sphere.py),
which is only ~300 lines of code.

For a brief intro to OpenFace, we provide face recognition with
a deep neural network that embed faces on a sphere.
(See [our tech report](http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf)
for a more detailed intro to how OpenFace works.)
Faces are often embedded onto a 128-dimensional sphere.
For this demo, we re-trained a neural network to embed faces onto a
3-dimensional sphere that we show in real-time on top of a camera feed.
The 3-dimensional embedding doesn't have the same accuracy as the
128-dimensional embedding, but it's sufficient to illustrate how
the embedding space distinguishes between different people.

In this demo:

+ We first use [OpenCV](http://opencv.org/) to get, process, and display
  a video feed from the camera.
+ The detected faces and embeddings for every face can be easily obtained with
  [dlib](http://blog.dlib.net/) and OpenFace with
  [a few lines of code](http://cmusatyalab.github.io/openface/usage/).
+ The color of the embedding is created by mapping the location of the
  face in the frame to be a number between 0 and 1 and then using
  a [matplotlib colormap](http://matplotlib.org/examples/color/colormaps_reference.html).
+ To keep all of the graphics on a single panel, we draw the sphere on
  top of the same OpenCV buffer as the video.
  [OpenCV only has 2D drawing primitives](http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html),
  so we [isometrically project](https://en.wikipedia.org/wiki/Isometric_projection)
  the points from the 3D sphere into 2D so we can use OpenCV's 2D drawing primitives.
+ Since the images from the video are noisy, the embeddings will jump around
  a lot of the sphere if not dampened.
  We smooth this out with
  [dlib's object tracker](http://blog.dlib.net/2015/02/dlib-1813-released.html)
  to track of a face's average (dampened) embedding throughout
  the video frames.
+ Face detection and recognition cause the 'low' frame rate.
  The frame rate could be improved by only doing detection and recognition
  every few frames and using face tracking (which is fast) in between to
  update the face locations.

### Running on your computer

To run this on your computer:

1. [Set up OpenFace](http://cmusatyalab.github.io/openface/setup/).
2. Download the 3D model from
   [here](http://openface-models.storage.cmusatyalab.org/nn4.small2.3d.v1.t7).
3. Run [demos/sphere.py](https://github.com/cmusatyalab/openface/blob/master/demos/sphere.py)
   with the `--networkModel` argument pointing to the 3D model.
