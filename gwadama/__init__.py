"""GWADAMA

Gravitational Wave Dataset Management.

Collection of ad-hoc classes and functions intended to automate all
dataset-related operations:
- Wave generation
- Data input
- Train/Test splitting
- Injections
- Pre-processing (normalization, resampling, whitening, filtering, ...)
- Export to several formats.


DEPENDENCIES
------------
bilby
clawdia
gwpy
h5py
numpy
pandas
scipy
sklearn
tqdm
watpy
yaml

"""
from .datasets import *
from . import ioo
from . import synthetic
