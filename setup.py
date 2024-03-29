from setuptools import find_packages, setup
from glob import glob


classes = """
    Development Status :: 4 - Beta
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Categorical Variational Autoencoders.')


setup(name='catvae',
      version='0.0.2',
      license='BSD-3-Clause',
      description=description,
      author_email="jamietmorton@gmail.com",
      maintainer_email="jamietmorton@gmail.com",
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'biom-format',
          'pandas>=1.0.0',
          'torch>=1.8.0',
          'gneiss',
          'geotorch>=0.1.0',
          'scikit-bio',
          'tensorboard',
          'pytorch-lightning>=1.3.1'
          # if utilizing the qiime2 API you also need a mininum of
          # https://github.com/qiime2/qiime2.git
          # https://github.com/qiime2/q2-sample-classifier.git
          # https://github.com/qiime2/q2-types.git
      ],
      classifiers=classifiers,
      package_data={},
      scripts=glob('scripts/*'))
