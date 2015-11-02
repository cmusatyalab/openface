# Demo 2: Comparing two images
The [comparison demo](https://github.com/cmusatyalab/openface/blob/master/demos/compare.py) outputs the predicted similarity
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
| <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-1.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-2.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/clapton-1.jpg' width='200px'></img> | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/clapton-2.jpg' width='200px'></img> |

The following table shows that a distance threshold of `0.5` would
distinguish these two people.
In practice, further experimentation should be done on the distance threshold.
On our LFW experiments, the mean threshold across multiple
experiments is 0.71 &plusmn; 0.027,
see [accuracies.txt](https://github.com/cmusatyalab/openface/blob/master/evaluation/lfw.nn4.v1.epoch-177/accuracies.txt).

| Image 1 | Image 2 | Distance |
|---|---|---|
| Lennon 1 | Lennon 2 | 0.310 |
| Lennon 1 | Clapton 1 | 1.241 |
| Lennon 1 | Clapton 2 | 1.056 |
| Lennon 2 | Clapton 1 | 1.386 |
| Lennon 2 | Clapton 2 | 1.073 |
| Clapton 1 | Clapton 2 | 0.259 |
