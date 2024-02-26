"""datasets.py

TODO

There are two basic type of datasets, 'clean' and 'injected'. Clean datasets'
classes inherit from the Base class, while Injected classes inherit from the
BaseInjected class.

"""
from copy import deepcopy
from pathlib import Path
from typing import Callable

from gwpy.timeseries import TimeSeries
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import make_interp_spline as sp_make_interp_spline
from sklearn.model_selection import train_test_split

from . import ioo
from .detectors import project_et
from .synthetic import (NonwhiteGaussianNoise, sine_gaussian_waveform, gaussian_waveform,
                        ring_down_waveform)
from .tima import resample
from .units import *


__all__ = ['Base', 'BaseInjected', 'SyntheticWaves', 'InjectedSyntheticWaves',
           'CoReWaves']


class Base:
    """Base class for all datasets.

    It is designed to store strains as nested dictionaries, with each level's
    key identifying a class/property of the strain. Each individual strain is a
    1D NDArray containing the features.
    
    By default there are two basic levels:
        - Class; to group up strains in categories.
        - Id; An unique identifier for each strain, which must exist in the
          metadata DataFrame as Index.
    
    Extra depths can be added, and will be thought of as modifications of the
    same original strains from the upper identifier level.
    
    Atributes
    ----------
    classes: dict[str]
        Ordered dict of labels, one per class (category).
    
    strains: dict {class: {id: strain} }
        Strains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        The class key is the name of the class, a string which must exist in
        the 'classes' dictionary.
        The 'id' is an unique identifier for each strain, and must exist in the
        `self.metadata.index` column of the metadata DataFrame.
        Extra depths can be added as variations of each strain.
    
    times: dict {class: {id: time_points} }, optional
        Time samples associated with the strains, following the same structure.
        Useful when the sampling rate is variable or different between strains.
        If None, all strains are assumed to be constantly sampled to the
        sampling rate indicated by the 'sample_rate' attribute.
    
    labels: NDArray[int]
        Indices of the classes, one per waveform.
        Each one points its respective waveform inside 'strains' to its class
        in 'classes'. The order is that of the index of 'self.metadata', and
        coincides with the order of the strains inside 'self.strains' if
        unrolled to a flat list of arrays.
    
    metadata: pandas.DataFrame
        All parameters and data related to the strains.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains up to the second depth level (the id.).
    
    sample_rate: int, optional
        If the 'times' attribute is present, this value is ignored. Otherwise
        it is assumed all strains are constantly sampled to this value.
    
    Xtrain, Xtest: dict {id: strain}
        Train and test subsets randomly split using SKLearn train_test_split
        function with stratified labels.
        The key corresponds to the strain's index at 'self.metadata'.
    
    Ytrain, Ytest: NDArray[int]
        1D Array containing the labels in the same order as 'Xtrain' and
        'Xtest' respectively.
    
    Caveats
    -------
    - The additional depths in the strains nested dictionary can't be directly
      tracked by the metadata Dataframe.
    - If working with two polarizations, they can be stored with just an
      extra depth layer.
    
    """
    def __init__(self):
        """Overwrite when inheriting!"""

        raise NotImplementedError

        # Must be defined:
        #
        self.classes: dict[str] = None
        self.labels: np.ndarray[int] = None
        self.metadata: pd.DataFrame = None
        self.strains: dict = None
        self.random_seed: int = None  # SKlearn train_test_split doesn't accept a Generator yet.
        self.rng: np.random.Generator = None

        # Optional
        #
        self.sample_rate: int = None
        self.times: dict = None

        # Additional attributes used/set by the methods of this class:
        #
        self.max_length = self._find_max_length()
        # Train/Test subset views.
        #   Timeseries:
        self.Xtrain = None
        self.Xtest = None
        #   Labels:
        self.Ytrain = None
        self.Ytest = None

    def resample(self, sample_rate, verbose=False) -> None:
        # Set up the time points associated to each strain in case it is not
        # provided.
        #
        if self.times is None:
            if sample_rate == self.sample_rate:
                raise ValueError("trying to resample to the same sampling rate")
            if self.sample_rate is None:
                raise ValueError("neither time samples nor a global sampling rate were defined")
            
            times = self._gen_times()
        else:
            times = self.times

        for *keys, strain in self.items():
            time = _get_value_from_nested_dict(times, keys)
            strain_resampled, sf_up, factor_down = resample(strain, time, sample_rate, full_output=True)
            _set_value_to_nested_dict(self.strains, keys, strain_resampled)
            
            if verbose:
                print(
                    f"Strain {keys[0]}::{keys[1]} up. to {sf_up} Hz, down by factor {factor_down}"
                )

        self.sample_rate = sample_rate
    
    def get_strain(self, *indices) -> np.ndarray:
        """Get a single strain from the complete index coordinates.
        
        This is just a shortcut to avoid having to write several squared
        brackets.
        
        """        
        return _get_value_from_nested_dict(self.strains, indices)

    def keys(self) -> list:
        """Return the unrolled combinations of all strain identifiers.

        Return the unrolled combinations of all keys  of the nested dictionary
        of strains by a hierarchical recursive search.
        
        It can be thought of as the extended version of Python's
        'dict().keys()', although this returns a plain list.
        
        Returns
        -------
        keys: list
            The unrolled combination in a Python list.
        
        """
        keys = unroll_nested_dictionary_keys(self.strains)

        return keys

    def items(self):
        """Return a new view of the dataset's items with unrolled indices.

        Each iteration consists on a tuple containing all the nested keys in
        'self.strains' along with the corresponding strain,
        (clas, key, *, strain).
        
        It can be thought of as an extension of Python's `dict.items()`.
        Useful to quickly iterate over all items in the dataset.

        Example of usage with an arbitrary number of keys in the nested
        dictionary of strains:
        ```
        for *keys, strain in dataset.items():
            print(f"Number of identifiers: {len(keys)}")
            print(f"Length of the strain: {len(strain)}")
            do_something(strain)
        ```
        
        """
        for indices in self.keys():
            yield (*indices, self.get_strain(*indices))

    def _init_strains_dict(self) -> dict:
        return {clas: {} for clas in self.classes}

    def _find_max_length(self) -> int:
        """Return the length of the longest signal present in strains."""

        max_length = 0
        for clas, key in self.keys():
            l = self.strains[clas][key].shape[-1]
            if l > max_length:
                max_length = l

        return max_length

    def _gen_times(self) -> dict:
        """Generate the time arrays associated to the strains.

        Assumes a constant sampling rate.
        
        Returns
        -------
        times: dict
            Nested dictionary with the same shape as 'self.strains'.
        
        """
        times = self._init_strains_dict()  # might change this method's name
        for *keys, strain in self.items():
            length = len(strain)
            t_end = (length - 1) / self.sample_rate
            time = np.linspace(0, t_end, length)
            _set_value_to_nested_dict(times, keys, time)
        
        return times
    
    def _build_train_test_subsets(self):
        indices = np.arange(self.n_samples)  # keep track of samples after shuffle.
        
        i_train, i_test = train_test_split(
            indices,
            train_size=self.train_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=self.labels
        )

        self.Xtrain, self.Ytrain = self._build_subset_strains(i_train)
        self.Xtest, self.Ytest = self._build_subset_strains(i_test)
    
    def _build_subset_strains(self, indices):
        """Return a subset of strains and their labels based on an index list.
        
        The indices are w.r.t. the Pandas 'self.metadata' table.

        Returns
        -------
        strains: dict {id: strain}
            The id key is the strain's index at 'self.metadata'.
        
        labels: NDArray
            1D Array containing the labels in the same order as 'strains'.
        
        """
        strains = {}
        labels = np.empty(len(indices), dtype=int)
        for i, id_ in enumerate(indices):
            clas = self.metadata.at[id_, 'Class']
            strains[id_] = self.strains[clas][id_]
            labels[i] = self.labels[id_]
        
        return strains, labels

    def _update_train_test_subsets(self):
        """Builds again the Train/Test subsets from the main strains attribute."""

        i_train = list(self.Xtrain.keys())
        i_test = list(self.Xtest.keys())
        self.Xtrain, self.Ytrain = self._build_subset_strains(i_train)
        self.Xtest, self.Ytest = self._build_subset_strains(i_test)
        


class BaseInjected(Base):
    """Manage an injected dataset with multiple SNR values.

    - Single noise realization, the same for all injections.
    - Manage multiple SNR values.
    - Export all strains to individual GWFs.

    Atributes
    ---------
    TODO

    """
    def __init__(self,
                 clean_dataset: Base, *,
                 psd: np.ndarray | Callable,
                 detector: str,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 random_seed: int):
        """
        Relevant attributes are inherited from a "clean" dataset instance,
        which can be any inherited from BaseDataset whose strains have not
        been injected yet.

        If train/test subsets are present, they too are updated when performing
        injections or changing units, but only through re-building them from
        the main 'strains' attribute using the already generated indices.
        Original train/test subsets from the clean dataset are not inherited.
        
        WARNING: Initializing this class does not perform the injections! For
        that use the method 'gen_injections'.

        TODO: Parameters
        
        """
        # Inherit clean strain instance attributes.
        #
        self.classes = clean_dataset.classes.copy()
        self.labels = clean_dataset.labels.copy()
        self.metadata = deepcopy(clean_dataset.metadata)
        self.strains_clean = deepcopy(clean_dataset.strains)
        self.sample_rate = clean_dataset.sample_rate
        self.max_length = clean_dataset.max_length
        self.distance = clean_dataset.distance
        # Train/Test distribution indices.
        self.Itrain = np.asarray(clean_dataset.Xtrain.keys())
        self.Itest = np.asarray(clean_dataset.Xtest.keys())

        # Noise instance and related attributes.
        #
        self.psd, self._psd = self._setup_psd(psd)
        self.detector = detector
        self.freq_cutoff = freq_cutoff
        self.freq_butter_order = freq_butter_order
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.noise = self._generate_background_noise(noise_length)

        # Injection related.
        #
        self.strains = None
        self.snr_list = []
        self.pad = {}  # {snr: pad}
        # Train/Test subset views.
        #   Timeseries:
        self.Xtrain = None
        self.Xtest = None
        #   Labels:
        self.Ytrain = None
        self.Ytest = None
    
    def __getstate__(self):
        """Avoid error when trying to pickle PSD interpolator.
        
        Turns out pickle tries to serialize the PSD interpolant, however
        Pickle is not able to serialize encapsulated functions.
        
        """
        state = self.__dict__.copy()
        del state['psd']
        
        return state
    
    def __setstate__(self, state):
        """Avoid error when trying to pickle PSD interpolator.
        
        Turns out pickle tries to serialize the PSD interpolant, however
        Pickle is not able to serialize encapsulated functions.
        
        """
        psd, _ = self._setup_psd(state['_psd'])
        state['psd'] = psd
        self.__dict__.update(state)
    
    def _setup_psd(self, psd: np.ndarray | Callable) -> tuple[Callable, np.ndarray]:
        """Setup the PSD function or array depending on the input."""

        if callable(psd):
            psd_fun = psd
            # Compute a realization of the PSD function with 1 bin per
            # integer frequency.
            freqs = np.linspace(0, self.sample_rate//2, self.sample_rate//2)
            psd_array = np.stack([freqs, psd(freqs)])
            i_cut = np.argmin((freqs - self.freq_cutoff) < 0)
            psd_array[1,:i_cut] = 0
        else:
            # Build a spline quadratic interpolant for the input PSD array
            # which ensures to be 0 below the cutoff frequency.
            _psd_interp = sp_make_interp_spline(psd[0], psd[1], k=2)
            def psd_fun(freqs):
                psd = _psd_interp(freqs)
                i_cut = np.argmin((freqs - self.freq_cutoff) < 0)
                psd[:i_cut] = 0
                return psd
            psd_array = np.asarray(psd)
            
        return psd_fun, psd_array
    
    def _generate_background_noise(self, noise_length: int) -> 'NonwhiteGaussianNoise':
        """The noise realization is generated by NonwhiteGaussianNoise."""

        d: float = noise_length / self.sample_rate
        noise = NonwhiteGaussianNoise(
            duration=d, psd=self.psd, sample_rate=self.sample_rate,
            rng=self.rng, freq_cutoff=self.freq_cutoff
        )

        return noise
    
    def _init_strains_dict(self) -> dict[dict[dict]]:
        """Initializes the nested dictionary of strains.
        
        Initializes the nested dictionary of strains following the hierarchy
        in the metadata attribute.
        
        """
        return {clas: {key: {} for key in self.metadata[clas]} for clas in self.classes}
    
    def gen_injections(self, snr: int | float | list, pad: int = 0) -> None:
        """Inject all strains in simulated noise with the given SNR values.

        - The SNR is computed using a matched filter against the noise PSD.
        - If `pad > 0`, it also updates the time arrays.
        - If strain units are in geometrized, they will be converted first to
          IS, injected, and converted back to geometrized.
        - After each injection, applies a highpass filter at the low-cut
          frequency specified at __init__.
        
        Parameters
        ----------
        snr: int | float | list
        
        TODO
        
        """
        if isinstance(snr, (int, float)):
            snr_list = list(snr)
        elif isinstance(snr, list):
            snr_list = snr
        else:
            raise TypeError(f"'{type(snr)}' is not a valid 'snr' type")
        
        # In case is the 1st time making injections.
        if self.strains is None:
            self.strains = self._init_strains_dict()
        
        sr = self.sample_rate
        for clas, key in unroll_nested_dictionary_keys(self.strains_clean):
            gw_clean = self.strains_clean[clas][key]
            strain_clean_padded = np.pad(gw_clean[1], pad)
            
            for snr_ in snr_list:
                # Highpass filter to the clean signal.
                # NOTE: The noise realization is already generated without
                # frequency components lower than the cutoff (they are set to
                # 0 during the random sampling).
                strain_clean_padded = self.noise.highpass_filter(
                    strain_clean_padded, f_cut=self.freq_cutoff, f_order=self.freq_butter_order
                )
                injected, _ = self.noise.inject(strain_clean_padded, snr=snr_)
                self.strains[clas][key][snr_] = injected
        
        self._update_train_test_subsets()
        
        # Record new SNR values and related padding.
        self.snr_list += snr_list
        for snr_ in snr_list:
            self.pad[snr_] = pad
    
    def export_strains_to_gwf(self,
                              path: str,
                              channel: str,  # Name of the channel in which to save strains in the GWFs.
                              t0_gps: float = 0,
                              verbose=False) -> None:
        """Export all strains to GWF format, one file per strain."""

        for indices in self.keys():
            times, strain = self.get_strain(*indices)
            ts = TimeSeries(
                data=strain,
                times=t0_gps + times,
                channel=channel
            )
            key = indices[1].replace(':', '_') + '_snr' + str(indices[2])
            fields = [
                self.detector,
                key,
                str(int(t0_gps)),
                str(int(ts.duration.value * 1000))  # In milliseconds
            ]
            file = Path(path) / ('-'.join(fields) + '.gwf')
            ts.write(file)

            if verbose:
                print("Strain exported to", file)


class SyntheticWaves(Base):
    """Class for building synthetically generated wavforms and background noise.

    Part of the datasets for the CLAWDIA main paper.
    The classes are hardcoded:
        SG: Sine Gaussian,
        G:  Gaussian,
        RD: Ring-Down.


    Atributes
    ----------
    classes: dict[str]
        Ordered dict of labels, one per class (category).
    
    strains: dict {class: {key: gw_strains} }
        Strains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        The class key is the name of the class, a string which must exist in
        the 'classes' attribute.
        The 'key' is an identifier of each strain.
        In this case it's just the global index ranging from 0 to 'self.n_samples'.
    
    labels: NDArray[int]
        Indices of the classes, one per waveform.
        Each one points its respective waveform inside 'strains' to its class
        in 'classes'. The order is that of the index of 'self.metadata', and
        coincides with the order of the strains inside 'self.strains' if
        unrolled to a flat list of arrays.
    
    metadata: pandas.DataFrame
        All parameters and data related to the strains.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains.

    n_samples: int
        Total number of waveforms (samples).
    
    units: str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.
    
    Xtrain, Xtest: dict {key: strain}
        Train and test subsets randomly split using SKLearn train_test_split
        function with stratified labels.
        The key corresponds to the strain's index at 'self.metadata'.
    
    Ytrain, Ytest: NDArray[int]
        1D Array containing the labels in the same order as 'Xtrain' and
        'Xtest' respectively.

    """
    classes = {'SG': 'Sine Gaussian', 'G': 'Gaussian', 'RD': 'Ring-Down'}
    n_classes = 3

    def __init__(self,
                 *,
                 n_samples_per_class: int,
                 wave_parameters_limits: dict,
                 max_length: int,
                 peak_time_max_length: float,
                 amp_threshold: float,
                 tukey_alpha: float,
                 sample_rate: int,
                 train_size: int | float, 
                 random_seed: int = None):
        """
        Parameters
        ----------
        n_samples_per_class: int
            Number of samples per class to produce.

        wave_parameters_limits: dict
            Min/Max limits of the waveforms' parameters, 9 in total.
            Keys:
            - mf0,   Mf0:   min/Max central frequency (SG and RD).
            - mQ,    MQ:    min/Max quality factor (SG and RD).
            - mhrss, Mhrss: min/Max sum squared amplitude of the wave.
            - mT,    MT:    min/Max duration (only G).
        
        max_length: int
            Maximum length of the waves. This parameter is used to generate the
            initial time array with which the waveforms are computed.
        
        peak_time_max_length: float
            Time of the peak of the envelope of the waves in the initial time
            array (built with 'max_length').
        
        amp_threshold: float
            Fraction w.r.t. the maximum absolute amplitude of the wave envelope
            below which to end the wave by shrinking the array and applying a
            windowing to the edges.
        
        tukey_alpha: float
            Alpha parameter (width) of the Tukey window applied to each wave to
            make sure their values end at the exact duration determined by either
            the duration parameter or the amplitude threshold.
        
        train_size: int | float
            If int, total number of samples to include in the train dataset.
            If float, fraction of the total samples to include in the train
            dataset.
            For more details see 'sklearn.model_selection.train_test_split'
            with the flag `stratified=True`.
        
        sample_rate: int
        
        random_seed: int, optional.
        
        """
        self.n_samples_per_class = n_samples_per_class
        self.n_samples = self.n_samples_per_class * self.n_classes
        self.sample_rate = sample_rate
        self.wave_parameters_limits = wave_parameters_limits
        self.max_length = max_length
        self.peak_time_max_length = peak_time_max_length
        self.tukey_alpha = tukey_alpha
        self.amp_threshold = amp_threshold
        self.train_size = train_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.labels = np.repeat(np.arange(self.n_classes), self.n_samples_per_class)

        self._gen_metadata()
        self._gen_dataset()
        self._build_train_test_subsets()

    def _gen_metadata(self):
        """Generate random metadata associated with each waveform."""

        classes_list = []
        f0s_list = []
        Q_list = []
        hrss_list = []
        duration_list = []  # Will be modified afterwards to take into account
                            # the amplitude threshold.
        for clas in self.classes:
            for _ in range(self.n_samples_per_class):
                # Need to pass 'self' explicitely since I'm calling the methods
                # inside a dictionary attribute. Python doesn't seem to
                # recognise them as the same class methods this way.
                f0, Q, hrss, duration = self._gen_parameters[clas](self)
                classes_list.append(clas)
                f0s_list.append(f0)
                Q_list.append(Q)
                hrss_list.append(hrss)
                duration_list.append(duration)  

        self.metadata = pd.DataFrame({
            'Class': classes_list,  # strings
            'f0': f0s_list,
            'Q': Q_list,
            'hrss': hrss_list,
            'duration': duration_list
        })

    def _gen_dataset(self):
        if self.metadata is None:
            raise AttributeError("'metadata' needs to be generated first!")

        self.strains = self._init_strains_dict()
        t_max = self.max_length/self.sample_rate - 1/self.sample_rate
        times = np.linspace(0, t_max, self.max_length)
        
        for i in range(self.n_samples):
            params = self.metadata.loc[i].to_dict()
            clas = params['Class']
            match clas:
                case 'SG':
                    self.strains[clas][i] = sine_gaussian_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        f0=self.metadata.at[i,'f0'],
                        Q=self.metadata.at[i,'Q'],
                        hrss=self.metadata.at[i,'hrss']
                    )
                case 'G':
                    self.strains[clas][i] = gaussian_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        hrss=self.metadata.at[i,'hrss'],
                        duration=self.metadata.at[i,'duration'],
                        amp_threshold=self.amp_threshold
                    )
                case 'RD':
                    self.strains[clas][i] = ring_down_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        f0=self.metadata.at[i,'f0'],
                        Q=self.metadata.at[i,'Q'],
                        hrss=self.metadata.at[i,'hrss']
                    )

        self._apply_threshold_windowing()
    
    def _random_log_uniform(self, min, max):
        """Returns a random number between [min, max] spaced logarithmically."""

        exponent = self.rng.uniform(np.log10(min), np.log10(max))
        random = 10**exponent

        return random
    
    def _random_log_int(self, min, max):
        """Returns a random integer between [min, max] spaced logarithmically."""

        return int(self._random_log_uniform(min, max))

    
    def _gen_parameters_sine_gaussian(self):
        """Generate random parameters for a single Sine Gaussian."""

        limits = self.wave_parameters_limits
        thres = self.amp_threshold
        f0   = self._random_log_int(limits['mf0'], limits['Mf0'])  # Central frequency
        Q    = self._random_log_int(limits['mQ'], limits['MQ']+1)  # Quality factor
        hrss = self._random_log_uniform(limits['mhrss'], limits['Mhrss'])
        duration = 2 * Q / (np.pi * f0) * np.sqrt(-np.log(thres))
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_gaussian(self):
        """Generate random parameters for a single Gaussian."""

        lims = self.wave_parameters_limits
        f0   = None  #  Casted to np.nan afterwards.
        Q    = None  #-/
        hrss = self._random_log_uniform(lims['mhrss'], lims['Mhrss'])
        duration = self._random_log_uniform(lims['mT'], lims['MT'])  # Duration
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_ring_down(self):
        """Generate random parameters for a single Ring-Down."""

        lims = self.wave_parameters_limits
        thres = self.amp_threshold
        f0   = self._random_log_int(lims['mf0'], lims['Mf0'])  # Central frequency
        Q    = self._random_log_int(lims['mQ'], lims['MQ']+1)  # Quality factor
        hrss = self._random_log_uniform(lims['mhrss'], lims['Mhrss'])
        duration = -np.sqrt(2) * Q / (np.pi * f0) * np.log(thres)
        
        return (f0, Q, hrss, duration)

    _gen_parameters = {
        'SG': _gen_parameters_sine_gaussian,
        'G': _gen_parameters_gaussian,
        'RD': _gen_parameters_ring_down
    }

    def _apply_threshold_windowing(self):
        """Shrink waves in the dataset and update its duration in the metadata.

        Shrink them according to their pre-computed duration in the metadata to
        avoid almost-but-not-zero edges, and correct those marginal durations
        longer than the window.

        """
        for i in range(self.n_samples):
            clas = self.metadata.at[i,'Class']
            duration = self.metadata.at[i,'duration']
            ref_length = int(duration * self.sample_rate)
            
            if clas == 'RD':
                # Ring-Down waves begin at the center. However we want to
                # emphasize their energetic beginning, therefore we will leave
                # a symmetric part before their start with zeros.
                i0 = self.max_length // 2 - ref_length
                i1 = i0 + 2*ref_length
            else:
                # SG and G are both centered.
                i0 = (self.max_length - ref_length) // 2
                i1 = self.max_length - i0

            new_lenght = i1 - i0
            if i0 < 0:
                new_lenght += i0
                i0 = 0
            if i1 > self.max_length:
                new_lenght -= i1 - self.max_length
                i1 = self.max_length

            window = sp.signal.windows.tukey(new_lenght, alpha=self.tukey_alpha)
            # Shrink and window
            self.strains[clas][i] = self.strains[clas][i][i0:i1] * window

            self.metadata.at[i,'duration'] = new_lenght / self.sample_rate


class InjectedSyntheticWaves(BaseInjected):
    """TODO
    
    """
    pass


class CoReWaves(Base):
    """Manage all operations needed to perform over a noiseless CoRe dataset.

    TODO: Update this shit, it's been left outdated while the rework of the
    base classes.

    Initial strains and metadata are expected to be obtained from a CoReManager
    instance.

    NOTE: By default this class treats as different classes (categories) each
    equation of state (EOS) present in the CoReManager instance.

    Workflow:
    - Load the strains from a CoreWaEasy instance, discarding or cropping those
      indicated with their respective arguments.
    - Resample.
    - Project onto the ET detector arms.
    - Change units and scale from geometrized to IS and vice versa.
    - Export the (latest version of) dataset to a HDF5.
    - Export the (latest version of) dataset to a GWF.

    Atributes
    ----------
    strains: dict {'eos': {'key': gw_strains} }
        Latest version of the strains.
    
    metadata: dict {'eos': {'key': {...} } }
        Metadata associated to each strain. Example:
        ```
        metadata[eos][key] = {
            'id': str,
            'mass': float,
            'mass_ratio': float,
            'eccentricity': float,
            'mass_starA': float,
            'mass_starB': float,
            'spin_starA': float,
            'spin_starB': float
        }
        ```
    
    units: str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.

    """
    def __init__(self,
                 cdb: ioo.CoReManager,
                 *,
                 classes: list[str],
                 discarded: set,
                 cropped: dict,
                 # Source:
                 distance: float, inclination: float, phi: float):
        """TODO"""

        self.classes = classes
        self.discarded = discarded
        self.cropped = cropped
        # Source parameters
        self.distance = distance
        self.inclination = inclination
        self.phi = phi

        # Note: I'm passing 'self.units' as return value just for consistence;
        # its initial value ('IS') could've been set here directly though.
        self.strains, self.metadata, self.units = self._get_strain_and_metadata(cdb)
        self.max_length = self._find_max_length()

        self.sample_rate = None  # Set up after resampling
    
    def _get_strain_and_metadata(self, cdb: ioo.CoReManager) -> dict[dict[np.ndarray]]:
        strains = self._init_strains_dict()
        metadata = self._init_strains_dict()
        units = 'IS'  # CoReManager.gen_strain() method's output.
        for eos in self.classes:
            # Get and filter out GW simulations.
            keys = set(cdb.filter_by('id_eos', eos).index)
            try:
                keys -= self.discarded[eos]
            except KeyError:
                pass  # No discards.
            keys = sorted(keys)  # IMPORTANT!!! Keep order to be able to trace back simulations.
            for key in keys:
                # CoRe Rh data (in IS units):
                times, h_plus, h_cros = cdb.gen_strain(
                    key, self.distance, self.inclination, self.phi
                )
                # Crop those indicated at the parameter file:
                try:
                    t0, t1 = self.cropped[eos][key]
                except KeyError:
                    t0, t1 = times[[0,-1]]  # Don't crop.
                i0 = np.argmin(np.abs(times-t0))
                i1 = np.argmin(np.abs(times-t1))
                strains[eos][key] = np.stack([times, h_plus, h_cros])[:,i0:i1]
                # Associated metadata:
                md = cdb.metadata.loc[key]
                metadata[eos][key] = {
                    'id': md['database_key'],
                    'mass': md['id_mass'],
                    'mass_ratio': md['id_mass_ratio'],
                    'eccentricity': md['id_eccentricity'],
                    'mass_starA': md['id_mass_starA'],
                    'mass_starB': md['id_mass_starB'],
                    'spin_starA': md['id_spin_starA'],
                    'spin_starB': md['id_spin_starB'],
                }
        
        return strains, metadata, units
    
    def project(self, *, detector: str, ra, dec, geo_time, psi):
        """Project strains into the ET detector at specified coordinates.
        
        Parameters
        ----------
        detector: str
            Possibilities are 'ET1', 'ET2', 'ET3' and 'all'.
        
        ra, dec: float
            Sky position in equatorial coordinates.
        
        geo_time: int | float
            Time of injection in GPS.
        
        psi: float
            Polarization angle.
        
        """
        project_pars = dict(ra=ra, dec=dec, geocent_time=geo_time, psi=psi)
        for clas, key in self.keys():
            times, hp, hc = self.strains[clas][key]
            projected = project_et(
                hp, hc, parameters=project_pars, sf=self.sample_rate, 
                nfft=2*self.sample_rate, detector=detector
            )
            self.strains[clas][key] = np.stack([times, projected])
    
    def convert_to_IS_units(self) -> None:
        """Convert strains from geometrized to IS units."""
        
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        for clas, key in self.keys():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'IS'
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert strains from IS to geometrized units."""
        
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")

        for clas, key in self.keys():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'geometrized'

    def export_strains_to_hdf5(self, file: str):
        """Save strains and their metadata to a HDF5 file.
        
        CAUTION! THIS DOES NOT SAVE THE ENTIRE INSTANCE!
        
        """
        ioo.save_to_hdf5(file=file, data=self.strains, metadata=self.metadata)
    
    def convert_to_IS_units(self) -> None:
        """Convert clean and injected strains from geometrized to IS units.
        
        Will raise an error if no injections are yet generated.
        
        """
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        self._convert_strain_clean_to_IS_units()
        self._convert_to_IS_units()
        self._update_train_test_subsets()

        self.units = 'IS'
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert clean and injected strains from IS to geometrized units.
        
        Will raise an error if no injections are yet generated.
        
        """
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")
        
        self._convert_strain_clean_to_scaled_geometrized_units()
        self._convert_to_scaled_geometrized_units()
        self._update_train_test_subsets()

        self.units = 'geometrized'
    
    def _convert_to_IS_units(self) -> None:
        for clas, key, snr in self.keys():
            mass = self.metadata[clas][key]['mass']
            self.strains[clas][key][snr] *=  mass * MSUN_MET / (self.distance * MPC_MET)
            if self.times is not None:
                self.times[clas][key][snr] *= mass * MSUN_SEC
    
    def _convert_to_scaled_geometrized_units(self) -> None:
        for clas, key, snr in self.keys():
            mass = self.metadata[clas][key]['mass']
            self.strains[clas][key][snr] /=  mass * MSUN_MET / (self.distance * MPC_MET)
            if self.times is not None:
                self.times[clas][key][snr] /= mass * MSUN_SEC
    
    def _convert_strain_clean_to_IS_units(self) -> None:
        for clas, key in self.keys():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains_clean[clas][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains_clean[clas][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_clean_to_scaled_geometrized_units(self) -> None:
        for clas, key in self.keys():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains_clean[clas][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains_clean[clas][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)




def unroll_nested_dictionary_keys(dictionary: dict) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    Useful to iterate over all keys of a nested dictionary without having to
    use multiple loops.

    Parameters
    ----------
    dictionary: dict
        Nested dictionary.
    
    Returns
    -------
    : list
        Unrolled combinations of all keys of the nested dictionary.
    
    """
    return _unroll_nested_dictionary_keys(dictionary)


def _unroll_nested_dictionary_keys(dictionary: dict, current_keys: list = None) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    This is the recurrent function. Use the main function.
    
    """
    if current_keys is None:
        current_keys = []

    unrolled_keys = []

    for key, value in dictionary.items():
        new_keys = current_keys + [key]

        if isinstance(value, dict):
            unrolled_keys += _unroll_nested_dictionary_keys(value, current_keys=new_keys)
        else:
            unrolled_keys.append(new_keys)

    return unrolled_keys


def _get_value_from_nested_dict(dict_, keys: list):
    value = dict_
    for key in keys:
        value = value[key]
    
    return value


def _set_value_to_nested_dict(dict_, keys, value):
        """Set a value to an arbitrarily-depth nested dictionary.
        
        Parameters
        ----------
        dict_: dict
            Nested dictionary.
        
        keys: iterable
            Sequence of keys necessary to get to the element inside the nested
            dictionary.
        
        value: Any
        
        """
        key = keys[0]
        element = dict_[key]
        if isinstance(element, dict):
            _set_value_to_nested_dict(element, keys[1:], value)
        else:
            dict_[key] = value
