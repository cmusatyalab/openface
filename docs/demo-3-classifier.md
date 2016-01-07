# Demo 3: Training a Classifier
OpenFace's core provides a feature extraction method to
obtain a low-dimensional representation of any face.
[demos/classifier.py](https://github.com/cmusatyalab/openface/blob/master/demos/classifier.py)
shows a demo of how these representations can be
used to create a face classifier.

There is a distinction between training the deep neural network (DNN)
model for feature representation
and training a model for classifying people with the DNN model.
This shows how to use a pre-trained DNN model to train and use
a classification model.

## Creating a Classification Model

### 1. Create raw image directory.
Create a directory for your raw images so that images from different
people are in different subdirectories. The names of the labels or
images do not matter, and each person can have a different amount of images.
The images should be formatted as `jpg` or `png` and have
a lowercase extension.

```
$ tree data/mydataset/raw
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png
```


### 2. Preprocess the raw images
Change `8` to however many
separate processes you want to run:
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align innerEyesAndBottomLip <path-to-aligned-data> --size 96 & done`.

### 3. Generate Representations
`./batch-represent/main.lua -outDir <feature-directory> -data <path-to-aligned-data>`
creates `reps.csv` and `labels.csv` in `<feature-directory>`.

### 4. Create the Classification Model
Use `./demos/classifier.py train <feature-directory>` to produce
the classification model which is an SVM saved to disk as
a Python pickle.

Training uses [scikit-learn](http://scikit-learn.org) to perform
a grid search over SVM parameters.
For 1000's of images, training the SVMs takes seconds.

## Classifying New Images
We have released a `celeb-classifier.nn4.v2.pkl` classification model
that is trained on about 6000 total images of the following people,
which are the people with the most images in our dataset.
Classifiers can be created with far less images per
person.

+ America Ferrera
+ Amy Adams
+ Anne Hathaway
+ Ben Stiller
+ Bradley Cooper
+ David Boreanaz
+ Emily Deschanel
+ Eva Longoria
+ Jon Hamm
+ Steve Carell

For an example, consider the following small set of images
the model has no knowledge of.
For an unknown person, a prediction still needs to be made, but
the confidence score is usually lower.

Run the classifier with:

```
./demos/classifier.py infer ./models/openface/celeb-classifier.nn4.v2.pkl images/examples/{carell,adams,lennon}*
```

| Person | Image | Prediction | Confidence |
|---|---|---|---|
| Carell | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/carell.jpg' width='200px'></img> | SteveCarell | 0.89 |
| Adams | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/adams.jpg' width='200px'></img> | AmyAdams | 0.99 |
| Lennon 1 (Unknown) | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-1.jpg' width='200px'></img> | SteveCarell | 0.47 |
| Lennon 2 (Unknown) | <img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-2.jpg' width='200px'></img> | DavidBoreanaz | 0.66 |

# Minimal Working Example to Extract Features

```
mkdir -p classify-test/raw/{lennon,clapton}
cp images/examples/lennon-* classify-test/raw/lennon
cp images/examples/clapton-* classify-test/raw/clapton
./util/align-dlib.py classify-test/raw align innerEyesAndBottomLip classify-test/aligned --size 96
./batch-represent/main.lua -outDir classify-test/features -data classify-test/aligned
...
nImgs:  4
Represent: 4/4
```
