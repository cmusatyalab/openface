# Visualizing representations with t-SNE
[t-SNE](http://lvdmaaten.github.io/tsne/) is a dimensionality
reduction technique that can be used to visualize the
128-dimensional features OpenFace produces.
The following shows the visualization of the three people
in the training and testing dataset with the most images.

**Training**

![](../../images/train-tsne.png)

**Testing**

![](../../images/val-tsne.png)

These can be generated with the following commands from the root
`openface` directory.

1. Install prerequisites as below.
2. Preprocess the raw `lfw` images, change `8` to however many
   separate processes you want to run:
   `for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align affine <path-to-aligned-data> --size 96 &; done`.
3. Generate representations with `./batch-represent/main.lua -outDir <feature-directory (to be created)> -model models/openface/nn4.v1.t7 -data <path-to-aligned-data>`
4. Generate t-SNE visualization with `./util/tsne.py <feature-directory> --names <name 1> ... <name n>`
   This creates `tsne.pdf` in `<feature-directory>`.

# Visualizing layer outputs
Visualizing the output feature maps of each layer
is sometimes helpful to understand what features
the network has learned to extract.
With faces, the locations of the eyes, nose, and
mouth should play an important role.

[demos/vis-outputs.lua](https://github.com/cmusatyalab/openface/blob/master/demos/vis-outputs.lua)
outputs the feature maps from an aligned image.
The following shows the first 39 filters of the
first convolutional layer on two images
of John Lennon.

![](../../images/nn4.v1.conv1.lennon-1.png)
![](../../images/nn4.v1.conv1.lennon-2.png)
