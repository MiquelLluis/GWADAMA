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

"""
from .datasets import *
from .ioo import CoReManager
from . import synthetic

__version__ = "0.2.0"
