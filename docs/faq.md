# How can I make OpenFace run faster?

1. Resize your images so that faces are approximately 100x100 pixels
  before running detection and alignment.

2. Compile dlib with AVX instructions, as discussed
  [here](http://dlib.net/face_landmark_detection_ex.cpp.html).
  Use the `-DUSE_AVX_INSTRUCTIONS=ON` in the first `cmake` command.
  If your architecture does not support AVX, try SSE4 or SSE2.
