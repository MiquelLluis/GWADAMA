"""GWADAMA

Gravitational Wave Dataset Management.

Collection of ad-hoc classes and functions intended to automate
gravitational-wave data-related operations:

- Wave generation.

- Data input.

- Train/Test splitting (smart index tracking for Cross-Validation tasks).

- Injections.

- Pre-processing (normalization, resampling, whitening, filtering, ...)

- Export to several formats.


DEPENDENCIES
------------

- bilby (2.3.0)

- clawdia

- gwpy (3.0.7)

- numpy (1.25.0)

- pandas (2.0.3)

- scipy (1.11.1)

- sklearn (1.2.2)

- tqdm (4.66.1)

- watpy (0.1.1)

In parenthesis are the version numbers they've been tested with.

"""
from .datasets import *
from .ioo import CoReManager
from . import synthetic

__version__ = "0.2.0"
