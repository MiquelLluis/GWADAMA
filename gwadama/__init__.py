"""GWADAMA

Gravitational Wave Dataset Management.

Intended to automate all the dataset building, train and test splitting,
signal generation, and pre-processing.


DEPENDENCIES
------------
bilby
clawdia
gwpy
h5py
numpy
pandas
scipy
yaml

"""
from .datasets import *
from . import ioo
from . import synthetic
