from distutils.core import setup

setup(
    name='openface',
    version='0.2.0',
    description="Face recognition with Google's FaceNet deep neural network.",
    url='https://github.com/cmusatyalab/openface',
    packages=['openface'],
    package_data={'openface': ['*.lua']},
)
