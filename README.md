# OpenFace

*Free and open source face recognition with
deep neural networks.*

[ ![Build Status] [travis-image] ] [travis]
[ ![Release] [release-image] ] [releases]
[ ![License] [license-image] ] [license]
[ ![Gitter] [gitter-image] ] [gitter]

[travis-image]: https://travis-ci.org/cmusatyalab/openface.png?branch=master
[travis]: http://travis-ci.org/cmusatyalab/openface

[release-image]: http://img.shields.io/badge/release-0.1.1-blue.svg?style=flat
[releases]: https://github.com/cmusatyalab/openface/releases

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

[gitter-image]: https://badges.gitter.im/Join%20Chat.svg
[gitter]: https://gitter.im/cmusatyalab/openface

---

+ Website: http://cmusatyalab.github.io/openface/
+ Join the
  [cmu-openface group](https://groups.google.com/forum/#!forum/cmu-openface)
  or the
  [gitter chat](https://gitter.im/cmusatyalab/openface)
  for discussions and installation issues.
+ Development discussions and bugs reports are on the
  [issue tracker](https://github.com/cmusatyalab/openface/issues).

---

This research was supported by the National Science Foundation (NSF)
under grant number CNS-1518865.  Additional support
was provided by the Intel Corporation, Google, Vodafone, NVIDIA, and the
Conklin Kistler family fund.  Any opinions, findings, conclusions or
recommendations expressed in this material are those of the authors
and should not be attributed to their employers or funding sources.

# Citations

The following is a [BibTeX](http://www.bibtex.org/)
and plaintext reference
for the OpenFace GitHub repository.
The reference may change in the future.
The BibTeX entry requires the `url` LaTeX package.

```
@misc{amos2015openface,
    title        = {{OpenFace: Face Recognition with Deep Neural Networks}},
    author       = {Amos, Brandon and Harkes, Jan and Pillai, Padmanabhan and Elgazzar, Khalid and Satyanarayanan, Mahadev},
    howpublished = {\url{http://github.com/cmusatyalab/openface}},
    note         = {Accessed: 2015-11-11}
}

Brandon Amos, Jan Harkes, Padmanabhan Pillai, Khalid Elgazzar,
and Mahadev Satyanarayanan.
OpenFace: Face Recognition with Deep Neural Networks.
http://github.com/cmusatyalab/openface.
Accessed: 2015-11-11.
```

# Licensing
The source code and trained models `nn4.v1.t7` and
`celeb-classifier.nn4.v1.t7` are copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | Modified | License
---|---|---|
[Atcold/torch-TripletEmbedding](https://github.com/Atcold/torch-TripletEmbedding) | No | MIT
[facebook/fbnn](https://github.com/facebook/fbnn) | Yes | BSD
