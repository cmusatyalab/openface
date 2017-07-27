# Usage

## [API Documentation](http://openface-api.readthedocs.org/en/latest/index.html)

## Example

See [the image comparison demo](https://github.com/cmusatyalab/openface/blob/master/demos/compare.py) for a complete example
written in Python using a naive Torch subprocess to process the faces.

```Python
import openface

# `args` are parsed command-line arguments.

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim, cuda=args.cuda)

# `img` is a numpy matrix containing the RGB pixels of the image.
bb = align.getLargestFaceBoundingBox(img)
alignedFace = align.align(args.imgDim, img, bb,
                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep1 = net.forward(alignedFace)

# `rep2` obtained similarly.
d = rep1 - rep2
distance = np.dot(d, d)
```
