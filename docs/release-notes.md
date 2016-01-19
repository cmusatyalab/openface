# Release Notes

## 0.2.0 (2016/01/19)
+ See [this blog post](http://bamos.github.io/2016/01/19/openface-0.2.0/)
  for an overview,
  [the GitHub Milestone](https://github.com/cmusatyalab/openface/milestones/v0.2.0)
  for a high-level issue summary.
+ Training improvements from resulting in an accuracy increase from **76.1% to 92.9%**,
  which are from Bartosz Ludwiczuk's ideas and implementations in
  [this mailing list thread](https://groups.google.com/forum/#!topic/cmu-openface/dcPh883T1rk).
  These improvements also reduce the training time from a week to a day.
+ Nearly halved execution time thanks to [Hervé Bredin's](http://herve.niderb.fr/)
  suggestions and sample code for image alignment in
  [Issue 50](https://github.com/cmusatyalab/openface/issues/50).
+ Hosted
  [Python API Documentation](http://openface-api.readthedocs.org/en/latest/index.html).
+ [Docker automated build](https://hub.docker.com/r/bamos/openface) online.
+ Initial automatic tests written in [tests](https://github.com/cmusatyalab/openface/tree/0.2.0/tests).
+ [Tests successfully passing](https://travis-ci.org/cmusatyalab/openface/branches)
  in the Docker automated build in Travis.
+ Add
  [util/profile-pipeline.py](https://github.com/cmusatyalab/openface/tree/0.2.0/util/profile-pipeline.py)
  to profile the overall execution time on a single image.

## 0.1.1 (2015/10/15)
+ Fix debug mode of NaiveDlib alignment.
+ Add
  [util/prune-dataset.py](https://github.com/cmusatyalab/openface/tree/0.1.1/util/prune-dataset.py)
  for dataset processing.
+ Correct Docker dependencies.

## 0.1.0 (2015/10/13)
+ Initial release.
