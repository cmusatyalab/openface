# Demo 3: Training a Classifier
OpenFace's core provides a feature extraction method to
obtain a low-dimensional representation of any face.
[demos/classifier.py](https://github.com/cmusatyalab/openface/blob/master/demos/classifier.py) shows a demo of
how these representations can be used to create a face classifier.

This is trained on about 6000 total images of the following people,
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

This demo uses [scikit-learn](http://scikit-learn.org) to perform
a grid search over SVM parameters.
For 1000's of images, training the SVMs takes seconds.
Our trained model obtains 87% accuracy on this set of data.
[models/get-models.sh](https://github.com/cmusatyalab/openface/blob/master/models/get-models.sh)
will automatically download this classifier and place
it in `models/openface/celeb-classifier.nn4.v1.pkl`.

For an example, consider the following small set of images
the model has no knowledge of.
For an unknown person, a prediction still needs to be made, but
the confidence score is usually lower.

Run the classifier on your images with:

```
./demos/classifier.py infer ./models/openface/celeb-classifier.nn4.v1.pkl ./your-image.png
```

| Person | Image | Prediction | Confidence |
|---|---|---|---|
| Carell | <img src='../../images/examples/carell.jpg' width='200px'></img> | SteveCarell | 0.78 |
| Adams | <img src='../../images/examples/adams.jpg' width='200px'></img> | AmyAdams | 0.87 |
| Lennon 1 (Unknown) | <img src='../../images/examples/lennon-1.jpg' width='200px'></img> | DavidBoreanaz | 0.28 |
| Lennon 2 (Unknown) | <img src='../../images/examples/lennon-2.jpg' width='200px'></img> | DavidBoreanaz | 0.56 |
