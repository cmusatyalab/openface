from distutils.core import setup
setup(name='openface',
      version='0.1.1',
      description="Face recognition with Google's FaceNet deep neural network.",
      url='https://github.com/yarnbasket/openface',
      packages=['openface', 'openface.alignment'],
      package_data={'openface': ['*.lua']},
)
