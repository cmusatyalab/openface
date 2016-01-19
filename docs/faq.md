# FAQ

## How much time does OpenFace take to process an image?

The execution time depends on the size of the input images.
The following results are from processing these example images
of John Lennon and Steve Carell, which are respectively sized
1050x1400px and 891x601px on an 8 core 3.70 GHz CPU.
The network processing time is significantly less on a GPU.

<img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-1.jpg' height='200px' />
<img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/carell.jpg' height='200px' />

More time is spent using the off-the-shelf face detector
than in the deep neural network!
The alignment cost is negligible.
These times are obtained from averaging 100 trials with
our [util/profile-pipeline.py](https://github.com/cmusatyalab/openface/blob/master/util/profile-pipeline.py)
script.
The standard deviations are low,
see [the raw data](/data/2016-01-19/execution-times.txt).

<img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/performance.png' />

## How can I make OpenFace run faster?

1. Resize your images so that faces are approximately 100x100 pixels
  before running detection and alignment.

2. Compile dlib with AVX instructions, as discussed
  [here](http://dlib.net/face_landmark_detection_ex.cpp.html).
  Use the `-DUSE_AVX_INSTRUCTIONS=ON` in the first `cmake` command.
  If your architecture does not support AVX, try SSE4 or SSE2.


## I'm getting an illegal instruction error in the pre-built Docker container.

This is unfortunately a result of building the Docker container
on one machine that compiles software with non-standard CPU flags
and creates illegal instructions on architectures that don't support
the additional CPU features.
Using the binaries from the pre-built container on a CPU that
doesn't support these features results in the illegal instruction error.
We try to prevent these as much as possible by building the images
inside of a Docker machine.
If you are still having these issues, please fall back to building
the image from scratch instead of pulling from Docker Hub.
You'll need to build the
[opencv-dlib-torch Dockerfile](https://github.com/cmusatyalab/openface/blob/master/opencv-dlib-torch.Dockerfile),
change the `FROM` part of the
[OpenFace Dockerfile](https://github.com/cmusatyalab/openface/blob/master/Dockerfile)
to your version,
then build the OpenFace Dockerfile.
