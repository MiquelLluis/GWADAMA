"""GWADAMA

Gravitational Wave Dataset Management.

Intended to automate all the dataset building, train and test splitting,
signal generation, and pre-processing.


DEPENDENCIES
------------
bilby
clawdia
corewaeasy
gwpy
h5py
numpy
scipy
yaml

"""
from .datasets import CleanDataset, InjectedDataset
from . import ioo
from . import synthetic
