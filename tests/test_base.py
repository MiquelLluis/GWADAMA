import pytest
import pandas as pd
import numpy as np

from gwadama.datasets import Base
from gwadama import dictools


class DummyBaseDataset(Base):
    def __init__(self, *, strains, times, metadata, sample_rate, random_seed,
                 padding, whitened, whiten_params):
        self.strains = strains
        classes = {'sine': 1, 'gauss': 2, 'burst': 3}
        self._check_classes_dict(classes)
        self.classes = classes
        self._gen_labels()  # sets `self.labels`
        self.metadata = metadata
        
        self._dict_depth = dictools.get_depth(self.strains)
        self.max_length = self._find_max_length()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self._track_times = times is not None

        #----------------------------------------------------------------------
        # Attributes whose values can be set up or otherwise left as follows.
        #----------------------------------------------------------------------

        # Optional padding record.
        self.padding = padding

        # Whitening related attributes.
        self.whitened = whitened
        self.whiten_params = whiten_params
        self.nonwhiten_strains = None  # No need to set it before whitening.

        # Time tracking related attributes.
        self.sample_rate = sample_rate
        self.times = times
        
        # Train/Test subset splits (views into the same 'self.strains').
        #   Timeseries:
        self.Xtrain: np.ndarray = None
        self.Xtest: np.ndarray = None
        #   Labels:
        self.Ytrain: np.ndarray = None
        self.Ytest: np.ndarray = None
        #   Indices (sorted as in train and test splits respectively):
        self.id_train: np.ndarray = None
        self.id_test: np.ndarray = None
        



@pytest.fixture
def base(raw_strains):
    strains, metadata = raw_strains
    return DummyBaseDataset(
        strains=strains,
        times=None,
        metadata=metadata,
        sample_rate=4096,
        random_seed=42,
        padding={},
        whitened=False,
        whiten_params={}
    )


def test_initialization(base):
    assert isinstance(base, Base)

    assert dictools.get_depth(base.strains) >= 2
    assert dictools.get_types(base.strains) == {np.ndarray}

    