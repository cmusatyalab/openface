# Setup
The following instructions are for Linux and OSX only.
Please contribute modifications and build instructions if you
are interested in running this on other operating systems.

+ We strongly recommend using the [Docker](https://www.docker.com/)
  container unless you are experienced with building
  Linux software from source.
+ In OSX, you may have to change the hashbangs
  from `python2` to `python`.
+ OpenFace has been tested in Ubuntu 14.04 and OSX 10.10
  and may not work well on other distributions.
  Please let us know of any challenges you had to overcome
  getting OpenFace to work on other distributions.

## Warning for architectures other than 64-bit x86
See [#42](https://github.com/cmusatyalab/openface/issues/42).

## Check out git submodules
Clone with `--recursive` or run `git submodule init && git submodule update`
after checking out.

## With Docker
This repo can be used as a container with
[Docker](https://www.docker.com/) for CPU mode.
Depending on your Docker configuration, you may need to
run the docker commands as root.

### Automated Docker Build
The quickest way to getting started is to use our pre-built
automated Docker build, which is available from
[bamos/openface](https://hub.docker.com/r/bamos/openface/).
This does not require or use a locally checked out copy of OpenFace.
To use on your images, share a directory between your
host and the Docker container.

```
docker pull bamos/openface
docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
cd /root/openface
./demos/compare.py images/examples/{lennon*,clapton*}
./demos/classifier.py infer models/openface/celeb-classifier.nn4.small2.v1.pkl ./images/examples/carell.jpg
./demos/web/start-servers.sh
```


### Building a Docker Container
This builds a Docker container from a locally checked out copy of OpenFace,
which will take about 2 hours on a modern machine.
Be sure you have checked out the git submodules.
Run the following commands from the `openface` directory.

```
docker build -t openface .
docker run -p 9000:9000 -p 8000:8000 -t -i openface /bin/bash
cd /root/openface
./run-tests.sh
./demos/compare.py images/examples/{lennon*,clapton*}
./demos/classifier.py infer models/openface/celeb-classifier.nn4.small2.v1.pkl ./images/examples/carell.jpg
./demos/web/start-servers.sh
```

### Docker in OSX
In OSX, follow the
[Docker Mac OSX Installation Guide](https://docs.docker.com/installation/mac/)
and start a docker machine and connect your shell to it
before trying to build the container.
In the simplest case, this can be done with:

```
docker-machine create --driver virtualbox --virtualbox-memory 4096 default
eval $(docker-machine env default)
```

#### Docker memory issues in OSX

Some users have reported the following silent Torch/Lua failure
when running `batch-represent` caused by an out of memory issue.

```
/root/torch/install/bin/luajit: /openface/batch-represent/dataset.lua:191: attempt to perform arithmetic on a nil value
```

If you're experiencing this, make sure you have created a Docker machine
with at least 4GB of memory with `--virtualbox-memory 4096`.

## By hand
Be sure you have checked out the submodules and downloaded the models as
described above.
See the
[Dockerfile](https://github.com/cmusatyalab/openface/blob/master/Dockerfile)
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
dlib can be installed from [pypi](https://pypi.python.org/pypi/dlib)
or built manually and depends on boost libraries.
Building dlib manually with
[AVX support](http://dlib.net/face_landmark_detection_ex.cpp.html)
provides higher performance.

To build manually, download
[dlib v18.16](https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2),
then run the following commands.
For the final command, make sure the directory is in your default
Python path, which can be found with `sys.path` in a Python interpreter.
In OSX, use `site-packages` instead of `dist-packages`.

```
mkdir -p ~/src
cd ~/src
tar xf dlib-18.16.tar.bz2
cd dlib-18.16/python_examples
mkdir build
cd build
cmake ../../tools/python
cmake --build . --config Release
sudo cp dlib.so /usr/local/lib/python2.7/dist-packages
```

At this point, you should be able to start your `python2`
interpreter and successfully run `import cv2; import dlib`.

In OSX, you may get a `Fatal Python error: PyThreadState_Get: no current thread`.
You may be able to resolve by rebuilding `python` and `boost-python`
as reported in [#21](https://github.com/cmusatyalab/openface/issues/21),
but please file a new issue with us or [dlib](https://github.com/davisking/dlib)
if you are unable to resolve this.

### Torch
Install [Torch](http://torch.ch) from the instructions on their website.
At this point, the command-line program `th` should
be available in your shell.
Install the dependencies with `luarocks install $NAME`,
where `$NAME` is as listed below.

+ [dpnn](https://github.com/nicholas-leonard/dpnn)
+ [nn](https://github.com/torch/nn)
+ [optim](https://github.com/torch/optim)
+ [csvigo](https://github.com/clementfarabet/lua---csv)
+ [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn)
  (only with CUDA)
+ [fblualib](https://github.com/facebook/fblualib)
  (only for [training a DNN](http://cmusatyalab.github.io/openface/training-new-models/))
+ [tds](https://github.com/torch/tds)
  (only for [training a DNN](http://cmusatyalab.github.io/openface/training-new-models/))
+ [torchx](https://github.com/nicholas-leonard/torchx)
  (only for [training a DNN](http://cmusatyalab.github.io/openface/training-new-models/))
+ [optnet](https://github.com/fmassa/optimize-net)
  (optional, only for [training a DNN](http://cmusatyalab.github.io/openface/training-new-models/))

These can all be installed with:

```
for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done
```

### OpenFace
In OSX, install `findutils` and `coreutils` with Brew or MacPorts for
the prefixed GNU variants `gfind` and `gwc`.
These are required for the commands to be compatible with
the Linux defaults of these commands.

From the root OpenFace directory,
install the Python dependencies with
`sudo python2 setup.py install`.

Run [models/get-models.sh](https://github.com/cmusatyalab/openface/blob/master/models/get-models.sh)
to download pre-trained OpenFace
models on the combined CASIA-WebFace and FaceScrub database.
This also downloads dlib's pre-trained model for face landmark detection.
This will incur about 200MB of network traffic.


# Installing open face using `conda` (OSX and GNU/Linux)

Step 1. Install `miniconda` with the following commands
* `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
* `bash Miniconda3-latest-Linux-x86_64.sh` and follow instructions.
* Add a Python 2.7 environment with: `conda create --name openface python=2.7`
* Activate the new env with: `source activate openface`

Step 2. Install dependencies
* Add the `conda-forge` channel with: `conda config --add channels conda-forge`
* `conda install opencv numpy pandas scipy scikit-learn scikit-image dlib txaio twisted autobahn OpenSSL pyopenssl imagehash service_identity`

Step 3. Install Torch and dependencies
* Deactivate the `openface` environment by opening a new terminal.
* `git clone https://github.com/torch/distro.git ~/torch --recursive`
* `cd ~/torch; bash install-deps;`
* `./install.sh`
* Execute the following to install the Torch deps `for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done`

Step 4. Install open face in `openface` environment using
* `source activate openface`
* `git clone https://github.com/cmusatyalab/openface.git ~/openface` 
* `cd openface` 
* `python setup.py install`
* Download `dlib`s models with: `./models/get-models.sh`

Open face is now installed. Test it with
* `./demos/classifier.py infer models/openface/celeb-classifier.nn4.small2.v1.pkl ./images/examples/carell.jpg`
