# Setup
The following instructions are for Linux and OSX only.
Please contribute modifications and build instructions if you
are interested in running this on other operating systems.

We strongly recommend using the [Docker](https://www.docker.com/)
container unless you are experienced with building
Linux software from source.

Also note that in OSX, you may have to change the hashbangs
from `python2` to `python`.

## Warning for architectures other than 64-bit x86
See [#42](https://github.com/cmusatyalab/openface/issues/42).

## Check out git submodules
Clone with `--recursive` or run `git submodule init && git submodule update`
after checking out.

## Download the models
Run [models/get-models.sh](https://github.com/cmusatyalab/openface/blob/master/models/get-models.sh)
to download pre-trained OpenFace
models on the combined CASIA-WebFace and FaceScrub database.
This also downloads dlib's pre-trained model for face landmark detection.
This will incur about 500MB of network traffic for the compressed
models that will decompress to about 1GB on disk.

Be sure the md5 checksums match the following.
Use `md5sum` in Linux and `md5` in OSX.

```
openface(master)$ md5sum models/{dlib/*.dat,openface/*.{pkl,t7}}
73fde5e05226548677a050913eed4e04  models/dlib/shape_predictor_68_face_landmarks.dat
c0675d57dc976df601b085f4af67ecb9  models/openface/celeb-classifier.nn4.v1.pkl
a59a5ec1938370cd401b257619848960  models/openface/nn4.v1.t7
```

## With Docker
This repo can be deployed as a container with [Docker](https://www.docker.com/)
for CPU mode.
Be sure you have checked out the submodules and downloaded
the models as described above.
Depending on your Docker configuration, you may need to
run the docker commands as root.

To use, place your images in `openface` on your host and
access them from the shared Docker directory.

```
docker build -t openface ./docker
docker run -p 9000:9000 -t -i -v $PWD:/openface openface /bin/bash
cd /openface
./demos/compare.py images/examples/{lennon*,clapton*}
```

### Docker in OSX
In OSX, follow the
[Docker Mac OSX Installation Guide](https://docs.docker.com/installation/mac/)
and start a docker machine and connect your shell to it
before trying to build the container.
In the simplest case, this can be done with:

```
docker-machine create --driver virtualbox default
eval $(docker-machine env default)
```

## By hand
Be sure you have checked out the submodules and downloaded the models as
described above.
See the
[Dockerfile](https://github.com/cmusatyalab/openface/blob/master/docker/Dockerfile)
as a reference.

This project uses `python2` because of the `opencv`
and `dlib` dependencies.
Install the packages the Dockerfile uses with your package manager.
With `pip2`, install `numpy`, `pandas`, `scipy`, `scikit-learn`, and `scikit-image`.

Next, manually install the following.

### OpenCV
Download [OpenCV 2.4.11](https://github.com/Itseez/opencv/archive/2.4.11.zip)
and follow their
[build instructions](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html).

### dlib
dlib can alternatively by installed from [pypi](https://pypi.python.org/pypi/dlib),
but might be slower than building manually because they are not
compiled with AVX support.

dlib requires boost libraries to be installed.

To build manually, start by
downloading
[dlib v18.16](https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2),
then:

```
mkdir -p ~/src
cd ~/src
tar xf dlib-18.16.tar.bz2
cd dlib-18.16/python_examples
mkdir build
cd build
cmake ../../tools/python
cmake --build . --config Release
cp dlib.so ..
```

At this point, you should be able to start your `python2`
interpreter and successfully run `import cv2; import dlib`.

In OSX, you may get a `Fatal Python error: PyThreadState_Get: no current thread`.
You may be able to resolve by rebuilding `python` and `boost-python`
as reported in [#21](https://github.com/cmusatyalab/openface/issues/21),
but please file a new issue with us or [dlib](https://github.com/davisking/dlib)
if you are unable to resolve this.

### Torch
Install [Torch](http://torch.ch) from the instructions on their website
and install the [dpnn](https://github.com/nicholas-leonard/dpnn)
and [nn](https://github.com/torch/nn) libraries with
`luarocks install dpnn` and `luarocks install nn`.

If you want CUDA support, also install
[cudnn.torch](https://github.com/soumith/cudnn.torch).

At this point, the command-line program `th` should
be available in your shell.
