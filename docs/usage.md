# Usage
## Existing Models
See [the image comparison demo](demos/compare.py) for a complete example
written in Python using a naive Torch subprocess to process the faces.

```Python
import openface
from openface.alignment import NaiveDlib # Depends on dlib.

# `args` are parsed command-line arguments.

align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

# `img` is a numpy matrix containing the RGB pixels of the image.
bb = align.getLargestFaceBoundingBox(img)
alignedFace = align.alignImg("affine", args.imgDim, img, bb)
rep1 = net.forwardImage(alignedFace)

# `rep2` obtained similarly.
d = rep1 - rep2
distance = np.dot(d, d)
```
