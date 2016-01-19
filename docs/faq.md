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
