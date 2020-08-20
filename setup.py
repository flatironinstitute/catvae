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
      version='0.0.1',
      license='BSD-3-Clause',
      description=description,
      author_email="jamietmorton@gmail.com",
      maintainer_email="jamietmorton@gmail.com",
      packages=find_packages(),
      install_requires=[
          'numpy==1.19.1',
          'scipy==1.5.2',
          'biom-format==2.1.8',
          'pandas==1.1.0',
          'torch==1.6.0',
          'seaborn==0.10.1',
          'jupyter',
          'gneiss==0.4.6',
          'scikit-bio==0.5.6',
          'tensorboard==2.1.1',
          'pytorch-lightning==0.9'
      ],
      classifiers=classifiers,
      package_data={},
      scripts=glob('scripts/*'),
)
