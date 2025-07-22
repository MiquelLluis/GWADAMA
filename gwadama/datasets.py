"""datasets.py

Main classes to manage GW datasets.

There are two basic type of datasets, clean and injected:

- Clean datasets' classes inherit from the Base class, extending their properties
  as needed.

- Injected datasets' classes inherit from the BaseInjected class, and
  optionally from other UserDefined(Base) classes.

Notes
-----
- TODO: The Base and BaseInjected couple should be more general, building from
  unlabeled data as in UnlabeledWaves. 

"""
from copy import deepcopy
import itertools
from typing import Callable
import warnings

# from gwpy.timeseries import TimeSeries  # Lazy import
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy as sp
from scipy.interpolate import make_interp_spline as sp_make_interp_spline
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import ioo
from . import detectors
from . import dictools
from . import fat
from . import synthetic
from . import tat
from .units import *


__all__ = ['Base', 'BaseInjected',
           'SyntheticWaves', 'InjectedSyntheticWaves',
           'UnlabeledWaves', 'InjectedUnlabeledWaves',
           'CoReWaves', 'InjectedCoReWaves']


class Base:
    """Base class for all datasets.

    TODO: Update docstring.

    Any dataset made of 'clean' (noiseless) GW must inherit this class.
    It is designed to store strains as nested dictionaries, with each level's
    key identifying a class/property of the strain. Each individual strain is a
    1D NDArray containing the features.
    
    By default there are two basic levels:
        
        - Class; to group up strains in categories.
        
        - Id; An unique identifier for each strain, which must exist in the
          metadata DataFrame as Index.
    
    Extra depths can be added, and will be thought of as modifications of the
    same original strains from the upper identifier level. If splitting the
    dataset into train and test susbsets, only combinations of (Class, Id) will
    be considered.

    Warning
    -------
        This class shall not be called directly. Use one of its subclasses.
    
    Attributes
    ----------
    classes : dict
        Dict of strings and their integer labels, one per class (category).

    metadata : pandas.DataFrame
        All parameters and data related to the strains.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains up to the second depth level (the ID).
        The total number of different waves must be equal to `len(metadata)`;
        this does not include possible variations such polarizations or
        multiple scallings of the same waveform when performing injections.
    
    strains : dict[dict [...]]
        Strains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        
        - Shape: {class: {id: strain} }
        
        - The 'class' key is the name of the class, which must exist in the
          'classes' list.
        
        - The 'id' is a unique identifier for each strain, and must exist in
          the index of the 'metadata' (DataFrame) attribute.
        
        - Extra depths can be added as variations of each strain, such as
          polarizations.
    
    labels : dict
        Class label of each wave ID, with shape {id: class_label}.
        Each ID points to the label of its class in the 'classes' attribute.
        Can be automatically constructed by calling the '_gen_labels()' method.
    
    max_length : int
        Length of the longest strain in the dataset.
        Remember to update it if modifying the strains length.

    padding : dict, optional
        Padding added to the strains with the form:
            {id: (pad_left, pad_right)}
        This only keeps track of any padding added for later potential usages.
    
    times : dict, optional
        Time samples associated with the strains, following the same structure
        up to the second depth level: {class: {id: time_points} }
        Useful when the sampling frequency is variable or different between strains.
        If None, all strains are assumed to be constantly sampled to the
        sampling frequency indicated by the 'fs' attribute, which must be
        provided.
    
    fs : int, optional
        If the 'times' attribute is present, this value is ignored. Otherwise
        it is assumed all strains are constantly sampled to this value.
        
        .. note::
            If dealing with variable sampling frequencies, avoid setting this
            attribute to anything other than None.
    
    random_seed : int, optional
        Seed used to initialize the random number generator (RNG), as well as
        for calling :func:`sklearn.model_selection.train_test_split` to
        generate the Train and Test subsets.
    
    Xtrain, Xtest : dict, optional
        Train and test subsets randomly split using SKLearn train_test_split
        function with stratified labels.
        Shape: {id: strain}.
        The 'id' corresponds to the strain's index at 'self.metadata'.
        They are just another views into the same data stored at 'self.strains',
        so no copies are performed.
    
    Ytrain, Ytest : NDArray[int], optional
        1D Array containing the labels in the same order as 'Xtrain' and
        'Xtest' respectively.
        See the attribute 'labels' for more info.

    id_train, id_test : NDArray[int], optional
        1D Array containing the id of the signals in the same order as
        'Xtrain' and 'Xtest' respectively.
    
    Notes
    -----
    - The additional depths in the strains nested dictionary can't be directly
      tracked by the metadata Dataframe.
    - If working with two polarizations, they can be stored with just an
      extra depth layer.
    - TODO: Always check self.times (when provided) to determine wether the
      sampling frequency is variable. Depending on the result, act accordingly with
      the current value of `self.fs`.
    
    """
    def __init__(self):
        """Overwrite when inheriting!"""
        raise NotImplementedError("Base class should not be called directly.")

        #----------------------------------------------------------------------
        # Attributes whose values must be set up during initialization.
        #----------------------------------------------------------------------
    
        self.strains: dict = None
        self.classes: dict[str] = None
        self._check_classes_dict(self.classes)
        self.metadata: pd.DataFrame = None
        self._gen_labels()  # sets `self.labels`
        
        # Number of nested layers in strains' dictionary. Keep updated always:
        self._dict_depth: int = dictools.get_depth(self.strains)

        self.max_length = self._find_max_length()
        self.random_seed: int = None  # SKlearn train_test_split doesn't accept a Generator yet.
        self.rng = np.random.default_rng(self.random_seed)
        self._track_times = False  # If True, self.times must be not None.

        #----------------------------------------------------------------------
        # Attributes whose values can be set up or otherwise left as follows.
        #----------------------------------------------------------------------

        # Optional padding record.
        self.padding = {}

        # Whitening related attributes.
        self.whitened = False
        self.whiten_params = {}
        self.nonwhiten_strains = self.strains  # Initially assumed to be the same.

        # Time tracking related attributes.
        self.fs: int = None
        self.times: dict = None
        
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
    
    def __str__(self):
        """Return a summary of the dataset."""
        #TODO: Add padding information
        
        # Get the name of the class
        class_name = self.__class__.__name__
        
        # Basic information
        num_classes = len(self.classes) if self.classes else 0
        num_strains = len(self) if self.strains else 0
        max_length = self.max_length if hasattr(self, 'max_length') else 0
        fs = self.fs if hasattr(self, 'fs') else None
        whitened = self.whitened if hasattr(self, 'whitened') else False
        train_test_split = (self.Xtrain is not None and self.Xtest is not None)
        
        # Metadata information
        metadata_shape = self.metadata.shape if hasattr(self, 'metadata') and self.metadata is not None else (0, 0)
        
        # Time tracking information
        time_tracking = self._track_times if hasattr(self, '_track_times') else False
        
        # Whitening information
        whitening_info = "Whitened" if whitened else "NOT whitened"
        
        # Train/Test split information
        split_info = "Performed" if train_test_split else "NOT performed"
        
        # Construct the string
        summary = [
            f"=== {class_name} Dataset Summary ===",
            f"  Classes: {num_classes}",
            f"  Strains: {num_strains}",
            f"  Max Strain Length: {max_length} samples",
            f"  Sampling frequency: {fs} Hz" if fs else "  Sampling frequency: Not specified",
            f"  Time Tracking: {'Enabled' if time_tracking else 'Disabled'}",
            f"  Whitening: {whitening_info}",
            f"  Train/Test Split: {split_info}",
            f"  Metadata Shape: {metadata_shape}",
            "=" * (len(class_name) + 24)  # Add a separator line matching the header length
        ]
        
        return "\n".join(summary)

    def __setstate__(self, state):
        """Convert legacy attribute name 'sample_rate' to 'fs'.
        
        It's possible that when loading with Pickle an old instance of this
        class, 'fs' was instead 'sample_rate', which was renamed in v0.4.0.

        """
        if 'sample_rate' in state:
            state['fs'] = state.pop('sample_rate')
        self.__dict__.update(state)

    def _check_classes_dict(self, classes: dict[str]):
        if not isinstance(classes, dict):
            raise TypeError("'classes' must be a dictionary")
        
        if not all(isinstance(k, (str,int)) for k in classes.keys()):
            raise TypeError("'classes' keys must be strings or integers")
        
        labels = classes.values()
        if not all(isinstance(label, int) for label in labels):
            raise TypeError("'classes' values must be integers")
        if len(set(labels)) != len(classes):
            raise ValueError("'classes' values must be unique")
    
    def __len__(self):
        """Return the total number of strains."""
        return dictools.get_number_of_elements(self.strains)

    def _gen_labels(self):
        """Generate the `self.labels` attribute

        The labels attribute maps each ID to the integer value of its class,
        mapped in the 'classes' attribute:
            {id: class_label} for each GW in the dataset
        
        """
        if hasattr(self, 'labels') and self.labels is not None:
            raise AttributeError("`labels` attribute already present.")

        self.labels = {}
        for clas, id_ in self.keys(max_depth=2):
            self.labels[id_] = self.classes[clas]

    def _gen_empty_strains_dict(self) -> dict:
        return {clas: {} for clas in self.classes}
    
    def _gen_empty_times_dict(self) -> dict:
        return dictools.replicate_structure(self.strains)

    def _find_max_length(self) -> int:
        """Return the length of the longest signal present in strains."""
        max_length = 0
        for *_, strain in self.items():
            l = len(strain)
            if l > max_length:
                max_length = l

        return max_length

    def _gen_times(self, t0=0):
        """Generate the time arrays associated to the strains.

        Generate the time arrays associated to the strains, assuming a constant
        sampling frequency. All time arrays begin at `t0`, 0 by default.
        
        """
        if self._track_times:
            raise RuntimeError(
                "Time arrays have already been generated. "
                "Check if '_track_times' was accidentally set or modify logic "
                "to avoid regenerating."
            )

        self.times = self._gen_empty_times_dict()
        for *keys, strain in self.items():
            times = tat.time_array_like(strain, fs=self.fs, t0=t0)
            dictools.set_value_to_nested_dict(self.times, keys, times)
        
        self._track_times = True

    def keys(self, max_depth: int = None) -> list:
        """Return the unrolled combinations of all strain identifiers.

        Return the unrolled combinations of all keys  of the nested dictionary
        of strains by a hierarchical recursive search.
        
        It can be thought of as the extended version of Python's
        'dict().keys()', although this returns a plain list.

        Parameters
        ----------
        max_depth : int, optional
            If specified, it is the number of layers to iterate to at most in
            the nested 'strains' dictionary.
        
        Returns
        -------
        keys : list
            The unrolled combination in a Python list.
        
        """
        keys = dictools.unroll_nested_dictionary_keys(self.strains, max_depth=max_depth)

        return keys

    def items(self):
        """Return a new view of the dataset's items with unrolled indices.

        Each iteration consists on a tuple containing all the nested keys in
        'self.strains' along with the corresponding strain,
        (clas, id, *, strain).
        
        It can be thought of as an extension of Python's `dict.items()`.
        Useful to quickly iterate over all items in the dataset.

        Example of usage with an arbitrary number of keys in the nested
        dictionary of strains:
        
        ```
        for *keys, strain in self.items():
            print(f"Number of identifiers: {len(keys)}")
            print(f"Length of the strain: {len(strain)}")
            do_something(strain)
        ```
        
        """
        for strain_indices in self.keys():
            yield (*strain_indices, self.get_strain(*strain_indices))

    def find_class(self, id):
        """Find which 'class' corresponds the strain 'id'.

        Finds the 'class' of the strain represented by the unique identifier
        'id'.

        Parameters
        ----------
        id : str
            Unique identifier of the string, that which also appears in the
            `metadata.index` DataFrame.
        
        Returns
        -------
        clas : int | str
            Class key associated to the strain 'id'.
        
        """
        return dictools.find_parent_key_of_nested_key(self.strains, id)

    def get_strain(self, *indices, normalize=False) -> np.ndarray:
        """Get a single strain from the complete index coordinates.
        
        This is just a shortcut to avoid having to write several squared
        brackets.

        NOTE: The returned strain is not a copy; if its contents are modified,
        the changes will be reflected inside the 'strains' attribute.

        Parameters
        ----------
        *indices : str | int
            The indices of the strain to retrieve.
        
        normalize : bool
            If True, the returned strain will be normalized to its maximum
            amplitude.
        
        Returns
        -------
        strain : np.ndarray
            The requested strain.
        
        """
        if len(indices) != self._dict_depth:
            raise ValueError("indices do not match the layout of 'self.strains'")

        strain = dictools.get_value_from_nested_dict(self.strains, indices)
        if normalize:
            strain /= np.max(np.abs(strain))

        return strain

    def get_strains_array(self, length: int = None) -> np.ndarray:
        """Get all strains stacked in a zero-padded Numpy 2d-array.

        Stacks all signals into an homogeneous numpy array whose length
        (axis=1) is determined by either 'length' or, if None, by the longest
        strain in the subset.
        The remaining space is zeroed.

        Parameters
        ----------
        length : int, optional
            Target length of the 'strains_array'. If None, the longest signal
            determines the length.

        Returns
        -------
        strains_array : np.ndarray
            train subset.
        
        lengths : list
            Original length of each strain, following the same order as the
            first axis of 'train_array'.

        """
        strains_flat = dictools.flatten_nested_dict(self.strains)
        strains_array, lengths = dictools.dict_to_stacked_array(strains_flat, target_length=length) 

        return strains_array, lengths

    def get_times(self, *indices) -> np.ndarray:
        """Get a single time array from the complete index coordinates.
        
        If there is no time tracking (thus no stored times), a new time array
        is generated using `self.fs` and the length of the
        correspoinding strain stored at the same index coordinates.

        .. warning::
            The returned array is not a copy; if its contents are modified,
            the changes will be reflected inside the 'times' attribute.
        
        """        
        if len(indices) != self._dict_depth:
            raise ValueError("indices do not match the layout of 'self.strains'")
        
        if self._track_times:
            times = dictools.get_value_from_nested_dict(self.times, indices)
        else:
            length = len(self.get_strain(*indices))
            duration = length / self.fs
            times = tat.gen_time_array(0, duration, self.fs, length=length)
        
        return times

    def _format_padding(self, padding) -> dict:
        """Format the padding into a dict of [left, right] padding per ID.

        This method standardizes the padding input into a dictionary mapping 
        each signal ID to a list of [left_pad, right_pad] values.
        
        Parameters
        ----------
        padding : int | ArrayLike | dict
            The padding specification. Allowed types:
        
            - **int**: Symmetrical padding (same left/right) for all signals.
            - **ArrayLike**: Sequence with exactly 2 elements interpreted as
              [left_pad, right_pad] for all signals.
            - **dict**: Pre-formatted dictionary with signal IDs as keys and
              [left, right] padding tuples as values. Returned directly without
              validation.

        Returns
        -------
        padding_dict : dict
            Dictionary mapping each signal ID to its [left_pad, right_pad]
            list.

        """
        if isinstance(padding, int):
            padding_dict = {id: [padding, padding] for id in self.labels}
        elif isinstance(padding, tuple|list|np.ndarray):
            padding_dict = {id: list(padding) for id in self.labels}
        elif isinstance(padding, dict):
            padding_dict = padding
        else:
            raise TypeError("padding must be an integer, an ArrayLike or a dictionary")
        
        return padding_dict

    def pad_strains(self, padding: int | ArrayLike | dict, window=None) -> None:
        """
        Pad strains with zeros on both sides.

        This function pads each strain with a specific number of samples on
        both sides. It also updates the 'max_length' attribute to reflect the
        new maximum length of the padded strains.

        Parameters
        ----------
        padding : int | ArrayLike | dict
            The padding to apply to each strain. If padding is an integer, it
            will be applied at both sides of all strains. If padding is a
            tuple, it must be of the form (left_pad, right_pad) in samples. If
            padding is a dictionary, it must be of the form {id: (left_pad,
            right_pad)}, where id is the identifier of each strain.

        window : str | tuple | Callable, optional
            Window to apply before padding the arrays.
            If str or tuple, it will be used a `scipy.signal.get_window(window)`.
            If Callable, it must take the strain before padding as argument,
            and return the windowed array.
            By default, no window is applied.

            .. versionadded:: 0.4.0
                This parameter was added in v0.4.0 to emphasize the potential
                need of windowing before padding strains to avoid spectral
                leakage.

        Notes
        -----
        - If time arrays are present, they are also padded accordingly.
        
        """
        padding = self._format_padding(padding)
        
        if window is None:
            window = lambda x: x  # identity
            warnings.warn(
                "No window is applied to the signal. This can cause issues "
                "when padding the signal, as it may introduce discontinuities. "
                "Consider using a windowing function."
            )
        elif isinstance(window, (str, tuple)):
            window = lambda x: sp.signal.get_window(window, x)
        elif not callable(window):
            raise TypeError(
                "window must be a Callable or a valid input for SciPy's "
                "`signal.get_window()` function."
            )

        for clas, id, *keys in self.keys():
            # Apply window if given
            strain = window(self.get_strain(clas, id, *keys))
            
            # Pad the strain
            left_pad, right_pad = padding[id]
            strain_padded = np.pad(strain, (left_pad, right_pad), mode='constant')
            dictools.set_value_to_nested_dict(self.strains, [clas, id, *keys], strain_padded)

            # Pad the corresponding time array if time tracking is enabled
            if self._track_times:
                times = self.get_times(clas, id, *keys)
                time_step = (times[-1] - times[0]) / (len(times) - 1)
                left_time_points = np.arange(times[0] - left_pad * time_step, times[0], time_step)
                right_time_points = np.arange(times[-1] + time_step, times[-1] + (right_pad + 1) * time_step, time_step)
                times_padded = np.concatenate([left_time_points, times, right_time_points])
                dictools.set_value_to_nested_dict(self.times, [clas, id, *keys], times_padded)

        if self.padding:
            # Add from previous padding the padded parts here.
            for id, pad_id in padding.items():
                self.padding[id] += pad_id
        else:
            self.padding = padding
        
        self.max_length = self._find_max_length()
        if self.Xtrain:
            self._update_train_test_subsets()

    def shrink_strains(self, padding: int | tuple | dict) -> None:
        """Shrink strains by a specified padding.

        Shrink strains (and their associated time arrays if present) by the
        specified padding, which is understood as negative.
        
        It also updates the 'max_length' attribute, and the previous padding
        if present.

        Parameters
        ----------
        padding : int | tuple | dict
            The pad to **shrink** to all strains. Values must be given in
            absolute value (positive int).
            If `pad` is an integer, symmettric shrinking is applied to all
            samples.
            If `pad` is a tuple, it must be of the form (pad_left, pad_right)
            in samples.
            If `pad` is a dictionary, it must be of the form
                {id: (pad_left, pad_right)},
            where id is the identifier of each
            strain.
            
            .. note::
                If extra layers below ID are present, they will be shrunk
                using the same pad in cascade.

        """
        padding = self._format_padding(padding)

        for clas, id, *keys in self.keys():
            strain = self.get_strain(clas, id, *keys)
            # Same shrinking limits for all possible strains below ID layer.
            pad_left, pad_right = padding[id]
            strain = strain[pad_left:-pad_right]
            dictools.set_value_to_nested_dict(self.strains, [clas,id,*keys], strain)

            if self._track_times:
                times = self.get_times(clas, id, *keys)
                times = times[pad_left:-pad_right]
                dictools.set_value_to_nested_dict(self.times, [clas,id,*keys], times)

        if self.padding:
            # Subtract from previous padding the shrunk parts here.
            for id, pad_id in padding.items():
                # THE SIGN IS APPLIED HERE.
                self.padding[id][0] -= pad_id[0]
                self.padding[id][1] -= pad_id[1]
        else:
            # If no previous pad wass added, store the current with negative
            # values (since we're shrinking, not enlarging).
            self.padding = {id: (-pad[0], -pad[1]) for id, pad in padding}
        
        self.max_length = self._find_max_length()
        if self.Xtrain:
            self._update_train_test_subsets()

    def resample(self, fs, verbose=False) -> None:
        """Resample strain and time arrays to a constant rate.

        This assumes time tracking either with time arrays or with the
        sampling frequency provided during initialization, which will be used to
        generate the time arrays previous to the resampling.

        This method updates the sampling frequency and the maximum length attributes.

        Parameters
        ----------
        fs : int
            The new sampling frequency in Hz.

        verbose : bool
            If True, print information about the resampling.
        
        """
        if fs == self.fs:
            raise ValueError("trying to resample to the same sampling frequency")
            
        if not self._track_times:
            self._gen_times()

        for *keys, strain in self.items():
            time = dictools.get_value_from_nested_dict(self.times, keys)
            strain_resampled, time_resampled, sr_interp, factor_up, factor_down = tat.resample(
                strain, time, fs, full_output=True
            )
            dictools.set_value_to_nested_dict(self.strains, keys, strain_resampled)
            dictools.set_value_to_nested_dict(self.times, keys, time_resampled)
            
            if verbose:
                print(
                    f"Strain {keys[0]}::{keys[1]} resampled {sr_interp} Hz → {fs} Hz (factors up/down: {factor_up}, {factor_down})"
                )

        self.fs = fs
        self.max_length = self._find_max_length()

    def bandpass(self,
                 *,
                 f_low: int | float,
                 f_high: int | float,
                 f_order: int | float,
                 verbose=False):
        """Apply a forward-backward digital bandpass filter.
        
        Apply a forward-backward digital bandpass filter to all clean strains
        between frequencies 'f_low' and 'f_high' with an order of 'f_order'.

        This method is intended to be used prior to any whitening.
        
        .. warning::
            This is an irreversible operation. Original (non-bandpassed)
            strains will be lost.
        
        """
        if self.whitened:
            raise RuntimeError("bandpass cannot be applied after whitening")

        if self.strains is None:
            raise RuntimeError("no strains have been given or generated yet")
        
        loop_aux = tqdm(self.items(), total=len(self)) if verbose else self.items()
        for *keys, strain in loop_aux:
            strain_filtered = fat.bandpass_filter(
                strain, f_low=f_low, f_high=f_high, f_order=f_order,
                fs=self.fs
            )
            # Update strains attribute.
            dictools.set_value_to_nested_dict(self.strains, keys, strain_filtered)

        if self.Xtrain is not None:
            self._update_train_test_subsets()
    
    def apply_window(self, window, all=False):
        """Apply a window to all strains.

        Apply a window to `self.strains` recursively, and optionally to
        `self.nonwhitened_strains` as well.

        Parameters
        ----------
        window : str | tuple
            Window to apply, formatted to be accepted by SciPy's `get_window`.

        all : bool, optional
            If True, apply the window also to `self.nonwhitened_strains`.

        Notes
        -----
        - Since strains may have different lengths, a window is generated for
          each one.
        - TODO: Generalise this method to BaseInjected for when `all=True`.
        
        """
        for *keys, strain in self.items():
            strain_windowed = strain * sp.signal.get_window(window, len(strain))
            dictools.set_value_to_nested_dict(self.strains, keys, strain_windowed)

        if all and (self.nonwhiten_strains is not None):
            for keys in dictools.unroll_nested_dictionary_keys(self.nonwhiten_strains):
                strain = dictools.get_value_from_nested_dict(
                    self.nonwhiten_strains,
                    keys
                )
                strain_windowed = strain * sp.signal.get_window(window, len(strain))
                dictools.set_value_to_nested_dict(
                    self.nonwhiten_strains,
                    keys,
                    strain_windowed
                )

    def normalise(self, mode='amplitude', all=False):
        """Normalise strains.

        Normalise strains to the indicated `mode`, and optionally to
        `self.nonwhitened_strains` as well.

        Parameters
        ----------
        mode : str, optional
            Normalisation method. Available: amplitude, l2

        all : bool, optional
            If True, normalise also `self.nonwhitened_strains`.

        Notes
        -----
        - TODO: Generalise this method to BaseInjected for when `all=True`.
        
        """
        if mode == 'amplitude':
            norm_coef_function = lambda x: 1/np.max(np.abs(x))
        elif mode == 'l2':
            norm_coef_function = lambda x: 1/np.linalg.norm(x)
        else:
            raise ValueError
        
        for *_, strain in self.items():
            strain[:] *= norm_coef_function(strain)
        
        if all and (self.nonwhiten_strains is not None):
            for keys in dictools.unroll_nested_dictionary_keys(self.nonwhiten_strains):
                strain = dictools.get_value_from_nested_dict(
                    self.nonwhiten_strains,
                    keys
                )
                strain[:] *= norm_coef_function(strain)
    
    def whiten(self,
               *,
               asd_array: np.ndarray,
               flength: int,
               highpass: int = None,
               normed=False,
               shrink: int = 0,
               window: str | tuple = 'hann',
               verbose=False):
        """Whiten the strains.
        
        Calling this method performs the whitening of all strains.
        
        Note
        ----
        Original (non-whitened) strains will be stored in the
        'nonwhiten_strains' attribute.
        
        """
        if self.whitened:
            raise RuntimeError("dataset already whitened")

        if self.strains is None:
            raise RuntimeError("no strains have been given or generated yet")
        
        self.nonwhiten_strains = deepcopy(self.strains)
        
        loop_aux = tqdm(self.items(), total=len(self)) if verbose else self.items()
        for *keys, strain in loop_aux:
            strain_w = tat.whiten(
                strain, asd=asd_array, fs=self.fs, flength=flength,
                highpass=highpass, normed=normed
            )
            # Update strains attribute.
            dictools.set_value_to_nested_dict(self.strains, keys, strain_w)
        
        if shrink > 0:
            self.shrink_strains(shrink)
            self.max_length = self._find_max_length()

        self.whitened = True
        self.whiten_params = {
            "asd_array": asd_array,  # Only saved in Base (clean).
            "flength": flength,
            "highpass": highpass,
            "normed": normed,
            "shrink": shrink,
            "window": window
        }

        if self.Xtrain is not None:
            self._update_train_test_subsets()

    def build_train_test_subsets(self, train_size: int | float):
        """Generate a random Train and Test subsets.

        Only indices in the 'labels' attribute are considered independent
        waveforms, any extra key (layer) in the 'strains' dict is treated
        monolithically during the shuffle.
        
        The strain values are just new views into the 'strains' attribute. The
        shuffling is performed by Scikit-Learn's function 'train_test_split',
        with stratification enabled.

        Parameters
        ----------
        train_size : int | float
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train subset.
            If int, represents the absolute number of train waves.
            
            Ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
            
        """            
        indices = list(self.labels)
        self.id_train, self.id_test = train_test_split(
            indices,
            train_size=train_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=list(self.labels.values())
        )
        self.Xtrain, self.Ytrain = self._build_subset_strains(self.id_train)
        self.Xtest, self.Ytest = self._build_subset_strains(self.id_test)        
    
    def _build_subset_strains(self, indices):
        """Return a subset of strains and their labels based on their ID.

        Return a new view into 'self.strains' using the input indices (ID) as
        the first layer of the nested dictionary.

        This collapses the first layer, the class, leaving the unique
        identifier ID as first layer. Nevertheless, the rest of possible layers
        beneath 'ID' are monolithically preserved.
        
        Parameters
        ----------
        indices : array-like
            The indices are w.r.t. 'self.labels'.

        Returns
        -------
        strains : dict {id: strain}
            The id key is the strain's index at 'self.metadata'.
        
        labels : NDArray
            1D Array containing the labels associated to 'strains'.
        
        """
        strains = {}
        labels = np.empty(len(indices), dtype=int)
        for i, id_ in enumerate(indices):
            labels[i] = self.labels[id_]
            clas = self.find_class(id_)
            strains[id_] = self.strains[clas][id_]
        
        return strains, labels

    def _update_train_test_subsets(self):
        """Builds again the Train/Test subsets from the main strains attribute.
        
        Each time the strains are **replaced** and mutability is not guaranteed
        to propagate changes to the train/test dictionaries, it is necessary to
        build them again, which is the purpose of this helper function.
        
        """
        id_train = list(self.Xtrain.keys())
        id_test = list(self.Xtest.keys())
        self.Xtrain, self.Ytrain = self._build_subset_strains(id_train)
        self.Xtest, self.Ytest = self._build_subset_strains(id_test)
    
    def get_xtrain_array(self, length=None, classes='all'):
        """Get the train subset stacked in a zero-padded Numpy 2d-array.

        Stacks all signals in the train subset into an homogeneous numpy array
        whose length (axis=1) is determined by either 'length' or, if None, by
        the longest strain in the subset. The remaining space is zeroed.

        Optionally, classes can be filtered by specifying which to include with
        the `classes` parameter.

        Parameters
        ----------
        length : int, optional
            Target length of the 'train_array'. If None, the longest signal
            determines the length.

        classes : str | List[str], optional
            Specify which classes to include. Include 'all' by default.

        Returns
        -------
        train_array : np.ndarray
            train subset.
        
        lengths : list
            Original length of each strain, following the same order as the
            first axis of 'train_array'.

        """
        train_subset = self.Xtrain.copy()

        if classes != 'all':
            class_labels = [self.classes[c] for c in classes]
            for class_int, id in zip(self.Ytrain, list(train_subset.keys())):
                if class_int not in class_labels:
                    del train_subset[id]

        return dictools.dict_to_stacked_array(train_subset, target_length=length)
    
    def get_xtest_array(self, length=None, classes='all'):
        """Get the test subset stacked in a zero-padded Numpy 2d-array.

        Stacks all signals in the test subset into an homogeneous numpy array
        whose length (axis=1) is determined by either 'length' or, if None, by
        the longest strain in the subset. The remaining space is zeroed.

        Optionally, classes can be filtered by specifying which to include with
        the `classes` parameter.

        Parameters
        ----------
        length : int, optional

        classes : str | List[str], optional
            Specify which classes to include. Include 'all' by default.

        Returns
        -------
        test_array : np.ndarray
            test subset.
        
        lengths : list
            Original length of each strain, following the same order as the
            first axis of 'test_array'.

        """
        test_subset = self.Xtest.copy()

        if classes != 'all':
            class_labels = [self.classes[c] for c in classes]
            for class_int, id in zip(self.Ytest, list(test_subset.keys())):
                if class_int not in class_labels:
                    del test_subset[id]

        return dictools.dict_to_stacked_array(test_subset, target_length=length)
    
    def get_ytrain_array(self, classes='all', with_id=False, with_index=False):
        """Get the filtered training labels.

        Parameters
        ----------
        classes : str | list[str] | 'all'
            The classes to include in the labels.
            All classes are included by default.

        with_id : bool
            If True, return also the list of related IDs.

        with_index : bool
            If True, return also the related GLOBAL indices; w.r.t. the stacked
            arrays returned by 'get_xtrain_array' WITHOUT filters.
            False by default.

        Returns
        -------
        np.ndarray
            Filtered train labels.

        np.ndarray, optional
            IDs associated to the filtered train labels.
        
        np.ndarray, optional
            Indices associated to the filtered train labels.

        """
        return self._filter_labels(
            self.Ytrain, list(self.Xtrain), classes,
            with_id=with_id, with_index=with_index
        )

    def get_ytest_array(self, classes='all', with_id=False, with_index=False):
        """Get the filtered test labels.

        Parameters
        ----------
        classes : str | list[str] | 'all'
            The classes to include in the labels.
            All classes are included by default.

        with_id : bool
            If True, return also the list of related IDs.

        with_index : bool
            If True, return also the related GLOBAL indices; w.r.t. the stacked
            arrays returned by 'get_xtest_array' WITHOUT filters.

        Returns
        -------
        np.ndarray
            Filtered test labels.

        np.ndarray, optional
            IDs associated to the filtered test labels.
        
        np.ndarray, optional
            Indices associated to the filtered test labels.

        """
        return self._filter_labels(
            self.Ytest, list(self.Xtest), classes,
            with_id=with_id, with_index=with_index
        )
    
    def _filter_labels(self, labels, labels_id, classes, with_id=False, with_index=False):
        """Filter labels based on 'classes'.

        This is a helper function for 'get_ytrain_array' and 'get_ytest_array'.
        
        Parameters
        ----------
        labels : np.ndarray
            The array containing the labels.
        
        labels_id : list
            IDs associated to the labels.
        
        classes : str | list[str] | 'all'
            The classes to include in the labels.
            All classes are included by default.
        
        with_id : bool
            If True, return also the related IDs.
            False by default.

        with_index : bool
            If True, return also the related indices w.r.t. the stacked array
            returned by '_stack_subset' given the strains related to 'labels'
            WITHOUT filters.
            False by default.

        Returns
        -------
        filtered_labels : np.ndarray
            Filtered labels.

        filtered_ids : np.ndarray, optional
            IDs associated to the filtered labels.

        filtered_indices : np.ndarray, optional
            Indices associated to the filtered labels.

        """
        if len(labels) != len(labels_id):
            raise ValueError("'labels' and 'labels_id' must have the same length.")

        if isinstance(classes, str):
            if classes == 'all':
                return (labels, labels_id) if with_id else labels
            else:
                classes = [classes]
        elif not isinstance(classes, list):
            raise TypeError("'classes' must be a string or list of strings.")
        
        filtered_labels = []
        filtered_ids = []
        filtered_indices = []
        
        i = 0
        for label, id in zip(labels, labels_id):
            if self.find_class(id) in classes:
                filtered_labels.append(label)
                filtered_ids.append(id)
                filtered_indices.append(i)
                i += 1  # Indices w.r.t. the FILTERED set!!!
        
        filtered_labels = np.array(filtered_labels)
        filtered_ids = np.array(filtered_ids)
        filtered_indices = np.array(filtered_indices)
        
        if with_id and with_index:
            return filtered_labels, filtered_ids, filtered_indices
        if with_id:
            return filtered_labels, filtered_ids
        if with_index:
            return filtered_labels, filtered_indices
        return filtered_labels
        
    def stack_by_id(self, id_list: list, length: int = None):
        """Stack an subset of strains by their ID into a Numpy array.

        Stack an arbitrary selection of strains by their original ID into a
        zero-padded 2d-array. The resulting order is the same as the order of
        that in 'id_list'.

        Parameters
        ----------
        id_list : list
            The IDs of the strains to be stacked.

        length : int, optional
            The target length of the stacked array. If None, the longest signal
            determines the length.

        Returns
        -------
        stacked_signals : np.ndarray
            The array containing the stacked strains.

        lengths : list
            The original lengths of each strain, following the same order as
            the first axis of 'stacked_signals'.

        Notes
        -----
        - Unlike in 'get_xtrain_array' and 'get_xtest_array', this method does
          not filter by 'classes' since it would be redundant, as IDs are
          unique.

        """
        if not isinstance(id_list, list):
            raise TypeError("'id_list' must be a list of IDs.")

        # Collapse the Class layer.
        strains = {id: ds for sub_strains in self.strains.values() for id, ds in sub_strains.items()}

        # Filter out those not in the 'id_list'.
        strains = dictools.filter_nested_dict(strains, lambda k: k in id_list, layer=0)
        assert len(strains) == len(id_list)

        # Sort them to match the order in 'id_list'.
        strains = {id: strains[id] for id in id_list}

        strains = dictools.flatten_nested_dict(strains)
        stacked_signals, lengths = dictools.dict_to_stacked_array(strains, target_length=length)
        
        return stacked_signals, lengths

        


class BaseInjected(Base):
    """Manage an injected dataset with multiple SNR values.

    It is designed to store strains as nested dictionaries, with each level's
    key identifying a class/property of the strain. Each individual strain is a
    1D NDArray containing the features.

    NOTE: Instances of this class or any other Class(BaseInjected) are
    initialized from an instance of any Class(Base) instance (clean dataset).
    
    By default there are THREE basic levels:
        
        - Class; to group up strains in categories.
        
        - Id; An unique identifier for each strain, which must exist in the
          metadata DataFrame as Index.
        
        - SNR; the signal-to-noise ratio at which has been injected w.r.t. a
          power spectral density of reference (e.g. the sensitivity of a GW
          detector).
    
    An extra depth can be added below, and will be treated as multiple
    injections at the same SNR value. This is usfeul for example to make
    injections at multiple noise realizations.


    Attributes
    ----------
    classes : list[str]
        List of labels, one per class (category).
    
    metadata : pandas.DataFrame
        All parameters and data related to the original strains, inherited
        (copied) from a clean Class(Base) instance.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains up to the second depth level (the ID).
        The total number of different waves must be equal to `len(metadata)`;
        this does not include possible variations such polarizations or
        multiple scallings of the same waveform when performing injections.
    
    strains_clean : dict[dict]
        Strains inherited (copied) from the `nonwhiten_strains` attribute of
        the Class(Base) instance.
        This copy is kept in order to perform new injections.
        
        - Shape: {class: {id: strain} }
        
        - The 'class' key is the name of the class, a string which must exist
          in the 'classes' list.
        
        - The 'id' is a unique identifier for each strain, and must exist in
          the index of the 'metadata' (DataFrame) attribute.
        
        .. warning::
            These strains should be not modified. If new clean strains are
            needed, create a new clean dataset instance first, and then
            initialise this class with it.
    
    strains : dict[dict]
        Injected trains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        
        - Shape: {class: {id: {snr: strain} } }
        
        - The 'class' key is the name of the class, a string which must exist
          in the 'classes' list.
        
        - The 'id' is a unique identifier for each strain, and must exist in
          the index of the 'metadata' (DataFrame) attribute.
        
        - The 'snr' key is an integer indicating the signal-to-noise ratio of
          the injection.

        - A fourth depth will be added below as additional injections per SNR
          if specified when performing the injections.
        
    labels : dict
        Indices of the class of each wave ID, inherited from a clean
        Class(Base) instance, with shape {id: class_index}.
        Each ID points to the index of its class in the 'classes' attribute.
    
    units : str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.

    times : dict, optional
        Time samples associated with the strains, following the same structure.
        Useful when the sampling frequency is variable or different between strains.
        If None, all strains are assumed to be constantly sampled to the
        sampling frequency indicated by the 'fs' attribute.
    
    fs : int
        Inherited from the parent Class(Base) instance.
    
    max_length : int
        Length of the longest strain in the dataset.
        Remember to update it if manually changing strains' length.
    
    random_seed : int
        Seed used to initialize the random number generator (RNG), as well as
        for calling :func:`sklearn.model_selection.train_test_split` to
        generate the Train and Test subsets.

    rng : np.random.Generator
        Random number generator used for sampling the background noise.
        Initialized with `np.random.default_rng(random_seed)`.

    detector : str
        GW detector name.

    psd_ : NDArray
        Numerical representation of the Power Spectral Density (PSD) of the
        detector's sensitivity.
    
    asd_ : NDArray
        Numerical representation of the Amplitude Spectral Density (ASD) of the
        detector's sensitivity.

    noise : gwadama.synthetic.NonwhiteGaussianNoise
        Background noise instance from NonwhiteGaussianNoise.

    snr_list : list
        List of SNR values at which each signal has been injected.

    pad : dict
        Padding introduced at each SNR injection, used in case the strains will
        be whitened after, to remove the vigneting at edges.
        It is associated to SNR values because the only implemented way to
        pad the signals is during the signal injection.

    injections_per_snr : int
        Number of injections per SNR value.
    
    injection_snr_scales : dict
        Scaling factors used for generating the injections, stored as a nested
        dictionary with the same structure as `self.strains`.
    
    whitened : bool
        Flat indicating whether the dataset has been whitened. Initially will
        be set to False, and changed to True after calling the 'whiten' method.
        Once whitened, this flag will remain True, since the whitening is
        implemented to be irreversible instance-wise.
    
    whiten_params : dict
        TODO

        freq_cutoff : int | float
            Frequency cutoff below which no noise bins will be generated in the
            frequency space, and also used for the high-pass filter applied to
            clean signals before injection.

        freq_butter_order : int
            Butterworth filter order.
            See (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
            for more information.

    Xtrain, Xtest : dict, optional
        Train and test subsets randomly split using SKLearn train_test_split
        function with stratified labels.
        Shape adds the SNR layer: {id: {snr: strain}}.
        The 'id' corresponds to the strain's index at 'self.metadata'.
    
    Ytrain, Ytest : NDArray[int], optional
        1D Array containing the labels in the same order as 'Xtrain' and
        'Xtest' respectively.
        
        .. warning::
            Does not include the SNR layer, therefore labels are not repeated.
    
    NOTES
    -----
    - TODO: Right now this class is oriented to simulate the background noise
      and apply the whitening using the same PSD and other parameters. This
      needs further generalization so that it can explicitly accept any
      pre-computed noise and a different PSD for whitening, as well as the
      possibility to estimate the PSD from the data in a programatically way.

    - TODO: Implement optioni in `gen_injections` to return (or store) the
      scaling factors.

    """
    def __init__(self,
                 clean_dataset: Base,
                 *,
                 psd: np.ndarray | Callable,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 noise_instance=None,
                 detector: str = '',
                 random_seed: int = None):
        """Base constructor for injected datasets.

        TODO: Update docstring.

        When inheriting from this class, it is recommended to run this method
        first in your __init__ function.

        Relevant attributes are inherited from the 'clean_dataset' instance,
        which can be any inherited from BaseDataset whose strains have not
        been injected yet.

        If train/test subsets are present, they too are updated when performing
        injections or changing units, but only through re-building them from
        the main 'strains' attribute using the already generated indices.
        Original train/test subsets from the clean dataset are not inherited.
        
        .. warning::
            Initializing this class does not perform the injections! For that use
            the method 'gen_injections'.

        Parameters
        ----------
        clean_dataset : Base
            Instance of a Class(Base) with noiseless signals.

        psd : np.ndarray | Callable
            Power Spectral Density of the detector's sensitivity in the range
            of frequencies of interest. Can be given as a callable function
            whose argument is expected to be an array of frequencies, or as a
            2d-array with shape (2, psd_length) so that:
            
            ```
            psd[0] = frequency_samples
            psd[1] = psd_samples
            ```
            
            .. note::
                `psd` is also used to compute the 'asd' attribute (ASD).

        noise_length : int
            Length of the background noise array to be generated for later use.
            It should be at least longer than the longest signal expected to be
            injected.

        freq_cutoff : int | float
            Frequency cutoff below which no noise bins will be generated in the
            frequency space, and also used for the high-pass filter applied to
            clean signals before injection.
            TODO: Properly separate this parameter from the whitening frequency
            cutoff, which can be set to a different value.

        freq_butter_order : int | float
            Butterworth filter order. For signals above 100 Hz it's usually
            enough with order 4 to 6.
            See (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
            for more information.

        noise_instance : NonwhiteGaussianNoise-like, optional
            [Experimental] Instead of generating random Gaussian noise, an
            already generated (or real) noise array can be given.

            .. warning::
                This option still needs to be properly integrated and tested.

        detector : str, optional
            GW detector name.
            Not used, just for identification.

        random_seed : int, optional
            Seed to initialize the random number generator (used for generating
            synthetic noise and injecting into random noise positions), as well
            as for calling :func:`sklearn.model_selection.train_test_split` to
            generate the Train and Test subsets.
        
        """
        if not clean_dataset.fs:
            raise ValueError("`fs` must be defined in order to perform injections")

        # Inherit clean strain instance attributes.
        #----------------------------------------------------------------------
        self.fs = clean_dataset.fs

        if clean_dataset.nonwhiten_strains is None:
            # Whitened space case (no access to strains before whitening).
            self._data_in_white_space = True
            self.strains_clean = deepcopy(clean_dataset.strains)
        else:
            # Non-whitened case (access to original strains).
            self._data_in_white_space = False
            self.strains_clean = deepcopy(clean_dataset.nonwhiten_strains)
        
        self.classes = clean_dataset.classes.copy()
        self._check_classes_dict(self.classes)
        self.labels = clean_dataset.labels.copy()
        self.metadata = deepcopy(clean_dataset.metadata)
        self._track_times = clean_dataset._track_times
        self.times = deepcopy(clean_dataset.times) if self._track_times else None
        self.padding = clean_dataset.padding.copy()
        self.max_length = clean_dataset.max_length

        # Noise instance and related attributes.
        #----------------------------------------------------------------------
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.detector = detector

        # Highpass parameters applied when generating the noise array.
        self.freq_cutoff = freq_cutoff
        self.freq_butter_order = freq_butter_order
    
        if self._data_in_white_space:
            self._psd, self.psd_array = None, None
            self._asd, self.asd_array = None, None
        else:
            self._psd, self.psd_array = self._setup_psd(psd)
            self._asd, self.asd_array = self._setup_asd_from_psd(psd)

        if noise_instance is None:
            # Generate synthetic non-white Guassian noise.
            if psd is None:
                raise ValueError(
                    "in order to generate synthetic background, 'psd' must be"
                    " provided."
                )
            self.noise = self._generate_background_noise(noise_length)
        else:
            # EXPERIMENTAL OPTION TO ALLOW THE USE OF REAL OR PRE-GENERATED
            # BACKGROUND NOISE.
            if not isinstance(noise_instance, synthetic.NonwhiteGaussianNoise):
                raise TypeError(
                    "'noise_instance' must be a valid noise type"
                    f" ({type(noise_instance)} was given)"
                )
            self.noise = noise_instance

        # Injection related:
        #----------------------------------------------------------------------
        self.strains = None
        self._dict_depth = clean_dataset._dict_depth + 1  # Depth of the strains dict.
        self.snr_list = []
        self.injection_snr_scales = None
        self.injections_per_snr = 1  # Default value.
        self.whitened = self._data_in_white_space
        self.whiten_params = None

        # Train/Test subset views:
        #----------------------------------------------------------------------
        if clean_dataset.Xtrain is not None:
            self.Xtrain = {k: None for k in clean_dataset.Xtrain.keys()}
            self.Xtest = {k: None for k in clean_dataset.Xtest.keys()}
            self.Ytrain = clean_dataset.Ytrain
            self.Ytest = clean_dataset.Ytest
            self.id_train = clean_dataset.id_train
            self.id_test = clean_dataset.id_test
        else:
            self.Xtrain = None
            self.Xtest = None
            self.Ytrain = None
            self.Ytest = None
            self.id_train = None
            self.id_test = None

    def __str__(self):
        """Return a summary of the dataset."""
        #TODO: Add padding information
        
        # Get the name of the class
        class_name = self.__class__.__name__
        
        # Basic information
        num_classes = len(self.classes) if self.classes else 0
        num_strains = len(self) if self.strains else 0
        max_length = self.max_length if hasattr(self, 'max_length') else 0
        fs = self.fs if hasattr(self, 'fs') else None
        whitened = self.whitened if hasattr(self, 'whitened') else False
        train_test_split = (self.Xtrain is not None and self.Xtest is not None)
        
        # Metadata information
        metadata_shape = self.metadata.shape if hasattr(self, 'metadata') and self.metadata is not None else (0, 0)
        
        # Time tracking information
        time_tracking = self._track_times if hasattr(self, '_track_times') else False
        
        # Whitening information
        whitening_info = "Whitened" if whitened else "NOT whitened"

        # Noise
        noise_length = len(self.noise)

        # Injections
        n_snr_injections = len(self.snr_list)
        
        # Train/Test split information
        split_info = "Performed" if train_test_split else "NOT performed"
        
        # Construct the string
        summary = [
            f"=== {class_name} Dataset Summary ===",
            f"  Classes: {num_classes}",
            f"  Strains: {num_strains}",
            f"  Max Strain Length: {max_length} samples",
            f"  Sampling frequency: {fs} Hz" if fs else "  Sampling frequency: Not specified",
            f"  Time Tracking: {'Enabled' if time_tracking else 'Disabled'}",
            f"  Whitening: {whitening_info}",
            f"  Noise realization lenght: {noise_length}",
            f"  Number of injections at different SNR: {n_snr_injections}",
            f"  Train/Test Split: {split_info}",
            f"  Metadata Shape: {metadata_shape}",
            "=" * (len(class_name) + 24)  # Add a separator line matching the header length
        ]
        
        return "\n".join(summary)
    
    def __getstate__(self):
        """Avoid error when trying to pickle PSD and ASD interpolants.
        
        Turns out Pickle tries to serialize the PSD and ASD interpolants,
        however Pickle is not able to serialize encapsulated functions.
        This is solved by removing said functions and computing the
        interpolants from their array representations when unpickling.

        NOTE: The loss of accuracy over repeated (de)serialization using this
        method has not been studied, use at your own discretion.
        
        """
        state = self.__dict__.copy()
        del state['_psd']
        del state['_asd']
        
        return state
    
    def __setstate__(self, state):
        """Avoid error when trying to unpickle PSD and ASD interpolants.
        
        Turns out Pickle tries to serialize the PSD and ASD interpolants,
        however Pickle is not able to serialize encapsulated functions.
        This is solved by removing said functions and computing the
        interpolants from their array representations when unpickling.

        Convert legacy attribute name 'sample_rate' to 'fs'.
        
        NOTE: The loss of accuracy over repeated (de)serialization using this
        method has not been studied, use at your own discretion.
        
        """
        if state['_data_in_white_space']:
            state['_psd'] = None
            state['_asd'] = None
        else:
            _psd, _ = self._setup_psd(state['psd_array'])
            _asd, _ = self._setup_asd_from_psd(state['psd_array'])
            state['_psd'] = _psd
            state['_asd'] = _asd
        if 'sample_rate' in state:
            state['fs'] = state.pop('sample_rate')
        self.__dict__.update(state)
    
    def _setup_psd(self, psd: np.ndarray | Callable) -> tuple[Callable, np.ndarray]:
        """Setup the PSD function or array depending on the input.
        
        Setup the power spectral density function and array from any of those.
        
        """
        if callable(psd):
            _check_vectorized(psd)
            psd_fun = psd
            # Compute a realization of the PSD function with 16 bins per
            # integer frequency to ensure the numerical representation has
            # enough precision.
            freqs = np.linspace(0, self.fs//2, self.fs*8)
            psd_array = np.stack([freqs, psd(freqs)])
        
        elif isinstance(psd, np.ndarray):
            # Build a spline quadratic interpolant for the input PSD array.
            psd_fun = sp_make_interp_spline(psd[0], psd[1], k=2)
            psd_array = np.asarray(psd)
        
        else:
            raise TypeError("'psd' type not recognized")
            
        return psd_fun, psd_array

    def _setup_asd_from_psd(self, psd) -> tuple[Callable, np.ndarray]:
        """Setup the ASD function or array depending on the input.
        
        Setup the amplitude spectral density function and array from any of
        those.
        
        """
        if callable(psd):
            asd_fun = lambda f: np.sqrt(psd(f))
            # Compute a realization of the ASD function with 16 bins per
            # integer frequency to ensure the numerical representation has
            # enough precision.
            freqs = np.linspace(0, self.fs//2, self.fs*8)
            asd_array = np.stack([freqs, asd_fun(freqs)])
        
        elif isinstance(psd, np.ndarray):
            # Build a spline quadratic interpolant for the input ASD array.
            asd_array = psd.copy()
            asd_array[1] = np.sqrt(psd[1])
            asd_fun = sp_make_interp_spline(asd_array[0], asd_array[1], k=2)
        
        else:
            raise TypeError("'psd' type not recognized")
            
        return asd_fun, asd_array

    def psd(self, frequencies: float | np.ndarray[float]) -> np.ndarray[float]:
        """Power spectral density (PSD) of the detector at given frequencies.

        Interpolates the PSD at the given frequencies from their array
        representation. If during initialization the PSD was given as its
        array representation, the interpolant is computed using SciPy's
        quadratic spline interpolant function.

        """
        return self._psd(frequencies)

    def asd(self, frequencies: float | np.ndarray[float]) -> np.ndarray[float]:
        """Amplitude spectral density (ASD) of the detector at given frequencies.

        Interpolates the ASD at the given frequencies from their array
        representation. If during initialization the ASD was given as its
        array representation, the interpolant is computed using SciPy's
        quadratic spline interpolant function.

        """
        return self._asd(frequencies)
    
    def _generate_background_noise(self, noise_length: int) -> synthetic.NonwhiteGaussianNoise:
        """The noise realization is generated by NonwhiteGaussianNoise."""
        d: float = noise_length / self.fs
        noise = synthetic.NonwhiteGaussianNoise(
            duration=d, psd=self.psd, fs=self.fs,
            rng=self.rng, freq_cutoff=self.freq_cutoff
        )

        return noise
    
    def _gen_empty_strains_dict(self) -> dict[dict[dict]]:
        """Initializes the nested dictionary of strains.
        
        Initializes the nested dictionary of strains following the hierarchy
        in the clean strains attribute, and adding the SNR layer.
        
        """
        strains_dict = dictools.replicate_structure(self.strains_clean)
        for indices in dictools.unroll_nested_dictionary_keys(strains_dict):
            dictools.set_value_to_nested_dict(strains_dict, indices, {})

        return strains_dict
    
    def gen_injections(self,
                       snr: int|float|list|tuple,
                       randomize_noise: bool = False,
                       random_seed: int = None,
                       injections_per_snr: int = 1,
                       verbose=False,
                       **inject_kwargs):
        """Inject all strains in simulated noise with the given SNR values.
        
        - The SNR is computed using a matched filter against the noise PSD.
        
        - If the strain is in geometrized units, it will be converted first to
          the IS, then injected and converted back to geometrized units.
        
        - The automatic highpass filter before each injection is not applied
          anymore. It is now assumed that clean signals are also filtered
          properly before.
        
        - If the method 'whiten' has been already called, all further
          injections will automatically be whitened with the same parameters,
          including the unpadding (if > 0).
        
        Parameters
        ----------
        snr : int | float | list | tuple

        randomize_noise : bool
            If True, the noise segment is randomly chosen before the injection.
            This can be used to avoid having the same noise injected for all
            clean strains.
            False by default.
            
            .. note::
                To avoid the possibility of repeating the same noise section
                in different injections, the noise realization must be
                reasonably large, e.g:
                
                `noise_length > n_clean_strains * self.max_length * len(snr)`
        
        random_seed : int, optional
            Random seed for noise realization, used only if `randomize_noise`
            is True.
            By default, the random number generator (RNG) created during
            initialization is used.
            
            .. warning::
                Setting this parameter creates a new RNG, replacing the one
                initialized with the class.  
                If this is unintended, do not provide this parameter.  
                A warning will be issued when it is used.

        injections_per_snr : int, optional
            Number of injections per SNR value. Defaults to 1.

            This is useful to minimize the statistical impact of the noise
            when performing injections at a sensitive (low) SNR.

        **inject_kwargs
            Additional arguments passed to the `_inject` method.
        
        Notes
        -----
        - If whitening is intended to be applied afterwards it is useful to
          pad the signals beforehand, in order to avoid the window vignetting
          produced by the whitening itself.
        
        - New injections are stored in the 'strains' atrribute.
        
        Raises
        ------
        ValueError
            Once injections have been performed at a certain SNR value, there
            cannot be injected again at the same value. Trying it will trigger
            this exception.
        
        """
        snr_list = self._validate_and_process_snr_input(snr)
        if set(snr_list) & set(self.snr_list):
            raise ValueError("one or more SNR values are already present in the dataset")

        times_old = deepcopy(self.times)
        if randomize_noise:
            self._setup_rng(random_seed)
        if verbose:
            n_injections = (
                dictools.get_number_of_elements(self.strains_clean)
                * len(snr_list)
                * injections_per_snr
            )
            pbar = tqdm(total=n_injections)
        
        self._initialize_injection_structures()

        self._perform_injections(randomize_noise, injections_per_snr, verbose,
                                 inject_kwargs, snr_list, times_old, pbar)

        if verbose:
            pbar.close()
        self.snr_list += snr_list
        self.injections_per_snr = injections_per_snr
        if injections_per_snr > 1:
            self._dict_depth = dictools.get_depth(self.strains)
        
        # Side-effect attributes updated.
        self.max_length = self._find_max_length()
        if self.Xtrain is not None:
            self._update_train_test_subsets()

    def _setup_rng(self, random_seed):
        """Set up or replace the RNG if necessary."""
        if random_seed is not None:
            if self.random_seed is not None:
                warnings.warn(
                        "Replacing the previous RNG by a new one with the provided "
                        f"random_seed = {random_seed}."
                    )
                # Replace the previous RNG by a new one.
            self.rng = np.random.default_rng(random_seed)

    def _initialize_injection_structures(self):
        """Initialize injection-related attributes."""
        if self.strains is None:
            # 1st time making injections.
            self.strains = self._gen_empty_strains_dict()
            if self._track_times:
                # Redo the dictionary structure to include the SNR layer.
                self.times = self._gen_empty_times_dict()
            self.injection_snr_scales = dictools.replicate_structure(self.strains)

    def _perform_injections(self, randomize_noise, injections_per_snr, verbose,
                            inject_kwargs, snr_list, times_old, pbar):
        """Main injection processing loop."""
        for clas, id_ in dictools.unroll_nested_dictionary_keys(self.strains_clean):
            # Clean signals are assumed to be already filtered if necessary.
            strain_clean = self.strains_clean[clas][id_]

            # Strain injections
            for snr_, rep in itertools.product(snr_list, range(injections_per_snr)):
                if randomize_noise:
                    pos0 = self.rng.integers(0, len(self.noise) - len(strain_clean))
                else:
                    pos0 = 0

                injected, scale = self._inject_signal_and_whiten(
                    strain_clean, snr_, id_, pos0, **inject_kwargs
                )
                
                # Save injected strains and scale factors.
                indices = [clas, id_, snr_]
                if injections_per_snr > 1:
                    indices.append(rep)
                dictools.set_value_to_nested_dict(self.strains, indices, injected, add_missing_keys=True)
                dictools.set_value_to_nested_dict(self.injection_snr_scales, indices, scale, add_missing_keys=True)

                if verbose:
                    pbar.update()
            
            # Update times.
            if self._track_times:
                # Make all SNR entries point to the SAME time array.
                # This keeps the shape of `self.times` consistent with strains
                # while avoiding unnecessary data duplication.
                times_i = times_old[clas][id_]  # TODO: This hardcoded indexing is not general enough!
                for snr_, rep in itertools.product(snr_list, range(injections_per_snr)):
                    indices = [clas, id_, snr_]
                    if injections_per_snr > 1:
                        indices.append(rep)
                    dictools.set_value_to_nested_dict(self.times, indices, times_i, add_missing_keys=True)

            # Shrink strains and times automatically if whitened was performed
            # alongside the injection. This will never occur when working in
            # the whitened space.
            if self.whitened and not self._data_in_white_space:
                shrink = self.whiten_params['shrink']
                if shrink:
                    self.shrink_strains(shrink)

    def _inject_signal_and_whiten(self, strain_clean, snr_, id_, pos0, **inject_kwargs):
        """Perform signal injection and optional whitening."""
        injected, scale = self._inject(
            strain_clean,
            snr_,
            id=id_,
            pos=pos0,
            **inject_kwargs
        )
        if self.whitened and not self._data_in_white_space:
            injected = tat.whiten(
                injected,
                asd=self.asd_array,
                fs=self.fs,
                flength=self.whiten_params['flength'],
                window=self.whiten_params['window'],
                highpass=self.whiten_params['highpass'],
                shrink=self.whiten_params['shrink'],
                normed=self.whiten_params['normed']
            )
            
        return injected, scale

    def _validate_and_process_snr_input(self, snr) -> list:
        if isinstance(snr, (int, float)):
            snr_list = [snr]
        elif isinstance(snr, (list,tuple)):
            snr_list = snr
        else:
            raise TypeError(f"'{type(snr)}' is not a valid 'snr' type")
        return snr_list
    
    def _inject(self,
                strain: np.ndarray,
                snr: int | float,
                pos: int = 0,
                **_) -> np.ndarray:
        """Inject 'strain' at 'snr' into noise using the 'self.noise' instance.

        NOTE: This is writen as an independent method to allow for other
        classes inheriting this to modify its behaviour without having to
        rewrite the entire 'gen_injections' method.

        Parameters
        ----------
        strain : NDArray
            Signal to be injected into noise.
        
        snr : int | float
            Signal to noise ratio.

        pos : int, optional
            Index position in the noise array where to inject the signal.
            0 by default.
        
        Returns
        -------
        injected : NDArray
            Injected signal.
        
        scale : float
            Scale factor applied to the signal.
        
        """
        injected, scale = self.noise.inject(strain, snr=snr, pos=pos)

        return injected, scale
    
    def export_strains_to_gwf(self,
                              path: str,
                              channel: str,  # Name of the channel in which to save strains in the GWFs.
                              t0_gps: float = 0,
                              verbose=False) -> None:
        """Export all strains to GWF format, one file per strain."""
        from pathlib import Path
        from gwpy.timeseries import TimeSeries

        for indices in self.keys():
            strain = self.get_strain(*indices)
            times = self.get_times(*indices)
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
    
    def whiten(self,
               *,
               flength: int,
               highpass: int = None,
               normed=False,
               shrink: int = 0,
               window: str | tuple = 'hann',
               verbose=False):
        """Whiten injected strains.
        
        Calling this method performs the whitening of all injected strains.
        Strains are later cut to their original size before adding the pad,
        to remove the vigneting.
        
        .. warning::
            This is an irreversible action; if the original injections need
            to be preserved it is advised to make a copy of the instance before
            performing the whitening.

        Parameters
        ----------
        flength : int
            Length (in samples) of the time-domain FIR whitening.
        
        highpass : float, optional
            Frequency cutoff.
        
        normed : bool
            Normalization applied after the whitening filter.

        shrink : int
            Margin at each side of the strain to crop (for each strain ID), in
            order to avoid edge effects. The corrupted area at each side is
            `0.5 * flength`, which corresponds to the amount of samples it
            takes for the whitening filter to settle.
        
        window : str | tuple, optional
            Window to apply to the strain prior to FFT, 'hann' by default.
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.
        
        """
        if self.whitened:
            raise RuntimeError("dataset already whitened")

        if self.strains is None:
            raise RuntimeError("no injections have been performed yet")
        
        loop_aux = tqdm(self.items(), total=len(self)) if verbose else self.items()
        for *keys, strain in loop_aux:
            strain_w = tat.whiten(
                strain, asd=self.asd_array, fs=self.fs,
                highpass=highpass, flength=flength, window=window, normed=normed
            )
            # Update strains attribute.
            dictools.set_value_to_nested_dict(self.strains, keys, strain_w)
        
        if shrink > 0:
            self.shrink_strains(shrink)
            self.max_length = self._find_max_length()

        self.whitened = True
        self.whiten_params = {
            'flength': flength,
            'highpass': highpass,
            'normed': normed,
            'shrink': shrink,
            'window': window
        }

        if self.Xtrain is not None:
            self._update_train_test_subsets()

    def get_xtrain_array(self,
                         length: int = None,
                         classes: str | list = 'all',
                         snr: int | list | str = 'all',
                         with_metadata: bool = False):
        """Get the train subset stacked in a zero-padded Numpy 2d-array.

        Stacks all signals in the train subset into an homogeneous numpy array
        whose length (axis=1) is determined by either 'length' or, if None, by
        the longest strain in the subset. The remaining space is zeroed.

        Allows the possibility to filter by class and SNR.

        NOTE: Same signals injected at different SNR are stacked continuously.

        Parameters
        ----------
        length : int, optional
            Target length of the 'train_array'. If None, the longest signal
            determines the length.

        classes : str | list[str]
            Whitelist of classes to include in the stack.
            All classes are included by default.

        snr : int | list[int] | str
            Whitelist of SNR injections to include in the stack. If more than
            one are selected, they are stacked zipped as follows:
            
            ```
            eos0 id0 snr0
            eos0 id0 snr1
                ...
            ```
            
            All injections are included by default.
        
        with_metadata : bool
            If True, the associated metadata is returned in addition to the
            train array in a Pandas DataFrame instance.
            This metadata is obtained from the original 'metadata' attribute,
            with the former index inserted as the first column, 'id', and with an
            additional column for the SNR values.
            False by default.

        Returns
        -------
        train_array : np.ndarray
            Train subset.
        
        lengths : list
            Original length of each strain, following the same order as the
            first axis of 'train_array'.
        
        metadata : pd.DataFrame, optional
            If 'with_metadata' is True, the associated metadata is returned
            with its entries in the same order as the 'train_array'.

        """
        return self._stack_subset(self.Xtrain, length, classes, snr, with_metadata)
    
    def get_xtest_array(self,
                        length: int = None,
                        classes: str | list = 'all',
                        snr: int | list | str = 'all',
                        with_metadata: bool = False):
        """Get the test subset stacked in a zero-padded Numpy 2d-array.

        Stacks all signals in the test subset into an homogeneous numpy array
        whose length (axis=1) is determined by either 'length' or, if None, by
        the longest strain in the subset. The remaining space is zeroed.

        Allows the possibility to filter by class and SNR.

        NOTE: Same signals injected at different SNR are stacked continuously.

        Parameters
        ----------
        length : int, optional
            Target length of the 'test_array'. If None, the longest signal
            determines the length.

        classes : str | list[str]
            Whitelist of classes to include in the stack.
            All classes are included by default.

        snr : int | list[int] | str
            Whitelist of SNR injections to include in the stack. If more than
            one are selected, they are stacked zipped as follows:
            
            ```
            eos0 id0 snr0
            eos0 id0 snr1
                ...
            ```
            
            All injections are included by default.

        with_metadata : bool
            If True, the associated metadata is returned in addition to the
            test array in a Pandas DataFrame instance.
            This metadata is obtained from the original 'metadata' attribute,
            with the former index inserted as the first column, 'id', and with an
            additional column for the SNR values.
            False by default.

        Returns
        -------
        test_array : np.ndarray
            Test subset.

        lengths : list
            Original length of each strain, following the same order as the
            first axis of 'test_array'.

        metadata : pd.DataFrame, optional
            If 'with_metadata' is True, the associated metadata is returned
            with its entries in the same order as the 'test_array'.
        
        """
        return self._stack_subset(self.Xtest, length=length, classes=classes,
                                  snr=snr, with_metadata=with_metadata)

    def _stack_subset(self,
                      strains: dict,
                      length:  int = None,
                      classes: str | list = 'all',
                      snr: int | list | str = 'all',
                      with_metadata: bool = False):
        """Stack 'strains' into a zero-padded 2d-array.

        This is a helper function for 'get_xtrain_array' and 'get_xtest_array'.

        Parameters
        ----------
        strains : dict
            A dictionary containing the strains to be stacked.
            The keys of the first layer are the IDs of the strains.

        length : int, optional
            The target length of the stacked array. If None, the longest signal
            determines the length.

        classes : str | list[str]
            Whitelist of classes to include in the stack.
            All classes are included by default.

        snr : int | list[int] | str
            Whitelist of SNR injections to include in the stack. If more than
            one are selected, they are stacked zipped as follows:
            
            ```
            eos0 id0 snr0
            eos0 id0 snr1
                ...
            ```
            
            All injections are included by default.

        with_metadata : bool
            If True, the associated metadata is returned in addition to the
            stacked array in a Pandas DataFrame instance.
            This metadata is obtained from the original 'metadata' attribute,
            with the former index inserted as the first column, 'id', and with
            an additional column for the SNR values.
            False by default.

        Returns
        -------
        stacked_signals : np.ndarray
            The array containing the stacked strains.

        lengths : list
            The original lengths of each strain, following the same order as
            the first axis of 'stacked_signals'.

        metadata : pd.DataFrame, optional
            If 'with_metadata' is True, the associated metadata is returned
            with its entries in the same order as the 'stacked_signals'.
            This metadata is obtained from the original 'metadata' attribute,
            with the former index inserted as the first column, 'id', and with
            an additional column for the SNR values.

        Raises
        ------
        ValueError
            If the value of 'classes' or 'snr' is not valid.

        """
        if isinstance(classes, (str, list)) and classes != 'all':
            if isinstance(classes, str):
                classes = [classes]

            # NOTE: Here there is no 'class' layer, therefore it must be
            # traced back from the ID, and filtered over this same layer.
            def filter_class(id):
                clas = self.find_class(id)
                return clas in classes
            strains = dictools.filter_nested_dict(strains, filter_class, layer=0)

        elif classes != 'all':
            raise TypeError("the type of 'classes' is not valid")


        if isinstance(snr, (int, list)):
            if isinstance(snr, int):
                snr = [snr]

            # NOTE: Here SNR is in Layer 1 because the Train/Test subset
            # dictionaries do not have the 'class' first layer.
            strains = dictools.filter_nested_dict(strains, lambda k: k in snr, layer=1)

        # If `snr == 'all'`, no filter is applied over 'strains'.
        elif isinstance(snr, str):
            if snr != 'all':
                raise ValueError("the value of 'snr' is not valid")
            else:
                pass
        
        else:
            raise TypeError("the type of 'snr' is not valid")
        

        strains = dictools.flatten_nested_dict(strains)

        stacked_signals, lengths = dictools.dict_to_stacked_array(strains, target_length=length)
        
        if with_metadata:
            id_list = [k[0] for k in strains]
            snr_list = [k[1] for k in strains]
            rep_list = [k[2] for k in strains]
            metadata = self.metadata.loc[id_list]  # sorts and makes all necessary copies.
            metadata.reset_index(inplace=True, names='id')
            metadata.insert(1, 'snr', snr_list)  # after 'id'.
            metadata.insert(2, 'rep', rep_list)  # after 'snr'.
            
            return stacked_signals, lengths, metadata
        return stacked_signals, lengths

    def get_ytrain_array(self, classes='all', snr='all', with_id=False, with_index=False):
        """Get the filtered training labels.

        Parameters
        ----------
        classes : str | list[str] | 'all'
            Whitelist of classes to include in the labels.
            All classes are included by default.
        
        snr : int | list[int] | str
            Whitelist of SNR injections to include in the labels.
            All injections are included by default.

        with_id : bool
            If True, return also the related IDs.
            False by default.

        with_index : bool
            If True, return also the related GLOBAL indices w.r.t. the stacked
            arrays returned by 'get_xtrain_array' WITHOUT filters.
            False by default.

        Returns
        -------
        np.ndarray
            Filtered train labels.

        np.ndarray, optional
            IDs associated to the filtered train labels.
        
        np.ndarray, optional
            Indices associated to the filtered train labels.

        """
        return self._filter_labels(
            self.Ytrain, list(self.Xtrain), classes, snr,
            with_id=with_id, with_index=with_index
        )

    def get_ytest_array(self, classes='all', snr='all', with_id=False, with_index=False):
        """Get the filtered test labels.

        Parameters
        ----------
        classes : str | list[str] | 'all'
            Whitelist of classes to include in the labels.
            All classes are included by default.
        
        snr : int | list[int] | str
            Whitelist of SNR injections to include in the labels.
            All injections are included by default.

        with_id : bool
            If True, return also the related IDs.
            False by default.

        with_index : bool
            If True, return also the related GLOBAL indices w.r.t. the stacked
            arrays returned by 'get_xtest_array' WITHOUT filters.

        Returns
        -------
        np.ndarray
            Filtered test labels.

        np.ndarray, optional
            IDs associated to the filtered test labels.
        
        np.ndarray, optional
            Indices associated to the filtered test labels.

        """
        return self._filter_labels(
            self.Ytest, list(self.Xtest), classes, snr,
            with_id=with_id, with_index=with_index
        )
    
    def _filter_labels(self, labels, labels_id, classes, snr, with_id=False, with_index=False):
        """Filter 'labels' based on 'classes' and 'snr'.

        This is a helper function for 'get_ytrain_array' and 'get_ytest_array'.
        
        Parameters
        ----------
        labels : np.ndarray
            The array containing the labels.

        labels_id : list
            IDs associated to the labels.
        
        classes : str | list[str] | 'all'
            Whitelist of classes to include in the labels.
            All classes are included by default.
        
        snr : int | list[int] | str
            Whitelist of SNR injections to include in the labels.
            All injections are included by default.

        with_id : bool
            If True, return also the related IDs.
            False by default.

        with_index : bool
            If True, return also the related indices w.r.t. the stacked array
            returned by '_stack_subset' given the strains related to 'labels'
            WITHOUT filters.
            False by default.

        Returns
        -------
        filtered_labels : np.ndarray
            Filtered labels.

        filtered_ids : np.ndarray, optional
            IDs associated to the filtered labels.

        filtered_indices : np.ndarray, optional
            Indices associated to the filtered labels.

        """
        # Get labels and IDs filtered by 'classes'.
        filtered_labels, filtered_ids, filtered_indices = super()._filter_labels(
            labels, labels_id, classes, with_id=True, with_index=True
        )

        if isinstance(snr, str):
            if snr != 'all':
                raise ValueError("only the str 'all' is allowed for 'snr'.")
        elif isinstance(snr, int):
            snr = [snr]
        elif not isinstance(snr, list):
            raise TypeError("the type of 'snr' is not valid")
        
        n_snr_total = len(self.snr_list)

        # Repeat all by the total number of SNR values.
        filtered_labels = np.repeat(filtered_labels, n_snr_total)
        filtered_ids = np.repeat(filtered_ids, n_snr_total)
        filtered_indices = np.repeat(filtered_indices, n_snr_total)

        n_filtered = len(filtered_labels)

        # Convert the indices to include the TOTAL number of SNR repetitions.
        for i in range(0, n_filtered, n_snr_total):
            i_old = filtered_indices[i]
            i_new0 = i_old * n_snr_total
            i_new1 = i_new0 + n_snr_total
            filtered_indices[i:i+n_snr_total] = np.arange(i_new0, i_new1)

        # Filter out those not present in the 'snr' list.
        if snr != 'all':
            mask = np.isin(self.snr_list, snr)
            mask = np.tile(mask, n_filtered//n_snr_total)
            filtered_labels = filtered_labels[mask]
            filtered_ids = filtered_ids[mask]
            filtered_indices = filtered_indices[mask]

        # Repeat labels and IDs by 'injections_per_snr', and extend the indices
        # accordingly.
        if self.injections_per_snr > 1:
            n_reps = self.injections_per_snr
            filtered_labels = np.repeat(filtered_labels, n_reps)
            filtered_ids = np.repeat(filtered_ids, n_reps)
            filtered_indices = np.repeat(filtered_indices, n_reps)
            # Convert the indices to also include the TOTAL number of
            # repetitions per SNR.
            for i in range(0, len(filtered_indices), n_reps):
                i_old = filtered_indices[i]
                i_new0 = i_old * n_reps
                i_new1 = i_new0 + n_reps
                filtered_indices[i:i+n_reps] = np.arange(i_new0, i_new1)

        if with_id and with_index:
            return filtered_labels, filtered_ids, filtered_indices
        if with_id:
            return filtered_labels, filtered_ids
        if with_index:
            return filtered_labels, filtered_indices
        return filtered_labels

    def stack_by_id(self,
                    id_list: list,
                    length: int = None,
                    snr_included: int | list[int] | str = 'all'):
        """Stack a subset of strains by ID into a zero-padded 2d-array.

        This may allow (for example) to group up strains by their original ID
        without leaking differnet injections (SNR) of the same strain into
        different splits.

        Parameters
        ----------
        id_list : array-like
            The IDs of the strains to be stacked.

        length : int, optional
            The target length of the stacked array. If None, the longest signal
            determines the length.

        snr_included : int | list[int] | str, optional
            The SNR injections to include in the stack. If more than one are
            selected, they are stacked zipped as follows:
            
            ```
            id0 snr0
            id0 snr1
              ...
            ```
            
            All injections are included by default.

        Returns
        -------
        stacked_signals : np.ndarray
            The array containing the stacked strains.

        lengths : list
            The original lengths of each strain, following the same order as
            the first axis of 'stacked_signals'.

        Notes
        -----
        - Unlike in 'get_xtrain_array' and 'get_xtest_array', this method does
          not filter by 'classes' since it would be redundant, as IDs are
          unique.

        Raises
        ------
        ValueError
            If the value of 'snr' is not valid.

        """
        if not isinstance(id_list, list):
            raise TypeError("'id_list' must be a list of IDs.")
        
        # Collapse the Class layer.
        strains = {id: ds for sub_strains in self.strains.values() for id, ds in sub_strains.items()}

        # Filter out those not in the 'id_list'.
        strains = dictools.filter_nested_dict(strains, lambda k: k in id_list, layer=0)

        # Filter out those injections whose SNR isnot in the 'snr' list.
        if isinstance(snr_included, (int, list)):
            if isinstance(snr_included, int):
                snr_included = [snr_included]
            # NOTE: Here SNR is in Layer 1 because we collapsed the Class layer.
            strains = dictools.filter_nested_dict(strains, lambda k: k in snr_included, layer=1)
        elif snr_included != 'all':
            raise ValueError("the value of 'snr' is not valid")

        strains = dictools.flatten_nested_dict(strains)  # keys: "(id, snr)"
        stacked_signals, lengths = dictools.dict_to_stacked_array(strains, target_length=length)
        
        return stacked_signals, lengths
        


class SyntheticWaves(Base):
    """Class for building synthetically generated wavforms and background noise.

    Part of the datasets for the CLAWDIA main paper.
    
    The classes are hardcoded:
        
        SG: Sine Gaussian,
        
        G:  Gaussian,
        
        RD: Ring-Down.


    Attributes
    ----------
    classes : dict
        Dict of strings and their integer labels, one per class (category).
    
    strains : dict {class: {key: gw_strains} }
        Strains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        The class key is the name of the class, a string which must exist in
        the 'classes' attribute.
        The 'key' is an identifier of each strain.
        In this case it's just the global index ranging from 0 to 'self.n_samples'.
    
    labels : NDArray[int]
        Indices of the classes, one per waveform.
        Each one points its respective waveform inside 'strains' to its class
        in 'classes'. The order is that of the index of 'self.metadata', and
        coincides with the order of the strains inside 'self.strains' if
        unrolled to a flat list of arrays.
    
    metadata : pandas.DataFrame
        All parameters and data related to the strains.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains.

    train_size : int | float
            If int, total number of samples to include in the train dataset.
            If float, fraction of the total samples to include in the train
            dataset.
            For more details see 'sklearn.model_selection.train_test_split'
            with the flag `stratified=True`.
    
    units : str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.
    
    Xtrain, Xtest : dict {key: strain}
        Train and test subsets randomly split using SKLearn train_test_split
        function with stratified labels.
        The key corresponds to the strain's index at 'self.metadata'.
    
    Ytrain, Ytest : NDArray[int]
        1D Array containing the labels in the same order as 'Xtrain' and
        'Xtest' respectively.

    """
    def __init__(self,
                 *,
                 classes: dict,
                 n_waves_per_class: int,
                 wave_parameters_limits: dict,
                 max_length: int,
                 peak_time_max_length: float,
                 amp_threshold: float,
                 tukey_alpha: float,
                 fs: int,
                 random_seed: int = None):
        """
        Parameters
        ----------
        n_waves_per_class : int
            Number of waves per class to produce.

        wave_parameters_limits : dict
            Min/Max limits of the waveforms' parameters, 9 in total.
            Keys:
            
            - mf0,   Mf0:   min/Max central frequency (SG and RD).
            
            - mQ,    MQ:    min/Max quality factor (SG and RD).
            
            - mhrss, Mhrss: min/Max sum squared amplitude of the wave.
            
            - mT,    MT:    min/Max duration (only G).
        
        max_length : int
            Maximum length of the waves. This parameter is used to generate the
            initial time array with which the waveforms are computed.
        
        peak_time_max_length : float
            Time of the peak of the envelope of the waves in the initial time
            array (built with 'max_length').
        
        amp_threshold : float
            Fraction w.r.t. the maximum absolute amplitude of the wave envelope
            below which to end the wave by shrinking the array and applying a
            windowing to the edges.
        
        tukey_alpha : float
            Alpha parameter (width) of the Tukey window applied to each wave to
            make sure their values end at the exact duration determined by either
            the duration parameter or the amplitude threshold.
        
        fs : int
        
        random_seed : int, optional.
        
        """
        self._check_classes_dict(classes)
        self.classes = classes
        self.n_waves_per_class = n_waves_per_class
        self.fs = fs
        self.wave_parameters_limits = wave_parameters_limits
        self.max_length = max_length
        self.peak_time_max_length = peak_time_max_length
        self.tukey_alpha = tukey_alpha
        self.amp_threshold = amp_threshold
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self._gen_metadata()
        self._track_times = False
        self._gen_dataset()
        self.nonwhiten_strains = self.strains
        self._gen_labels()

        self.Xtrain = None
        self.Xtest = None
        self.Ytrain = None
        self.Ytest = None

    def _gen_metadata(self):
        """Generate random metadata associated with each waveform."""
        classes_list = []
        f0s_list = []
        Q_list = []
        hrss_list = []
        duration_list = []  # Will be modified afterwards to take into account
                            # the amplitude threshold.
        for clas in self.classes:
            for _ in range(self.n_waves_per_class):
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
        """Generate the dataset from the previously generated metadata.

        After generating the waveforms with the analytical expressions it
        shrinks them to the specified duration in the metadata. This is
        necessary because the analytical expressions are infinite, so we apply
        a window to get perfect edges. However this does not necessary align
        with the exact duration provided by the metadata due to the signals
        being sampled at discrete values. Therefore after the windowing the
        final duration is computed again and updated in the metadata attribute.
        
        Attributes
        ----------
        strains : dict[dict]
            Creates the strains attribute with the properties stated at the
            class' docstring.
        
        _dict_depth : int
            Number of nested layers in strains' dictionary.
        
        metadata : pd.DataFrame
            Updates the duration of the waveforms after shrinking them.

        """
        if self.metadata is None:
            raise AttributeError("'metadata' needs to be generated first!")

        self.strains = self._gen_empty_strains_dict()

        t_max = (self.max_length - 1) / self.fs
        times = np.linspace(0, t_max, self.max_length)
        
        for id in range(len(self.metadata)):
            params = self.metadata.loc[id].to_dict()
            clas = params['Class']
            match clas:
                case 'SG':
                    self.strains[clas][id] = synthetic.sine_gaussian_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        f0=self.metadata.at[id,'f0'],
                        Q=self.metadata.at[id,'Q'],
                        hrss=self.metadata.at[id,'hrss']
                    )
                case 'G':
                    self.strains[clas][id] = synthetic.gaussian_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        hrss=self.metadata.at[id,'hrss'],
                        duration=self.metadata.at[id,'duration'],
                        amp_threshold=self.amp_threshold
                    )
                case 'RD':
                    self.strains[clas][id] = synthetic.ring_down_waveform(
                        times,
                        t0=self.peak_time_max_length,
                        f0=self.metadata.at[id,'f0'],
                        Q=self.metadata.at[id,'Q'],
                        hrss=self.metadata.at[id,'hrss']
                    )
        
        self._dict_depth = dictools.get_depth(self.strains)

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
        for i in range(len(self)):
            clas = self.metadata.at[i,'Class']
            duration = self.metadata.at[i,'duration']
            ref_length = int(duration * self.fs)
            
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

            self.metadata.at[i,'duration'] = new_lenght / self.fs


class InjectedSyntheticWaves(BaseInjected):
    """TODO
    
    """
    def __init__(self,
                 clean_dataset: SyntheticWaves,
                 *,
                 psd: np.ndarray | Callable,
                 detector: str,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 random_seed: int):
        super().__init__(
            clean_dataset, psd=psd, detector=detector, noise_length=noise_length,
            freq_cutoff=freq_cutoff, freq_butter_order=freq_butter_order, random_seed=random_seed
        )

        # Initialize the Train/Test subsets inheriting the indices of the input
        # clean dataset instance.
        if clean_dataset.Xtrain is not None:
            self.Xtrain = dictools.replicate_structure(clean_dataset.Xtrain)
            self.Xtest = dictools.replicate_structure(clean_dataset.Xtest)
            self.Ytrain = dictools.replicate_structure(clean_dataset.Ytrain)
            self.Ytest = dictools.replicate_structure(clean_dataset.Ytest)
        else:
            self.Xtrain = None
            self.Xtest = None
            self.Ytrain = None
            self.Ytest = None


class UnlabeledBaseMixin:
    """Mixin class for managing methods related to unlabeled datasets.

    This mixin modifies the functionality of the base class to handle datasets
    without associated class labels.
    
    This class is intended to be used as part of a dataset class that operates
    on unlabeled gravitational wave signals, ensuring compatibility with 
    methods from the base class while maintaining flexibility for unlabeled
    data.

    Notes
    -----
    - When inheriting this class, it should precede the base classes in the MRO
      to handle correctly the references.
    
    """
    def get_strain(self, *indices, normalize=False):
        # Add the dummy class name (if ommited) as the first index, so that the
        # user does not need to write it explicitly:
        class_label = next(iter(self.classes.keys()))
        if indices[0] != class_label:
            indices = (next(iter(self.classes.keys())), *indices)

        return super().get_strain(*indices, normalize=normalize)
    get_strain.__doc__ = Base.get_strain.__doc__

    def get_times(self, *indices) -> np.ndarray:
        # Add the dummy class name (if ommited) as the first index, so that the
        # user does not need to write it explicitly:
        class_label = next(iter(self.classes.keys()))
        if indices[0] != class_label:
            indices = (next(iter(self.classes.keys())), *indices)
        
        return super().get_times(*indices)
    get_times.__doc__ = Base.get_times.__doc__


class UnlabeledWaves(UnlabeledBaseMixin, Base):
    """Dataset class for clean gravitational wave signals without labels.

    This class extends `Base`, modifying its behavior to handle datasets 
    where gravitational wave signals are provided without associated labels.
    Unlike `Base`, it does not require a classification structure but 
    retains methods for loading, storing, and managing waveform data.

    The dataset consists of nested dictionaries, storing each waveform in an 
    independent array to accommodate variable lengths.

    Attributes
    ----------    
    strains : dict
        Dictionary of stored waveforms, indexed by unique identifiers.
    
    max_length : int
        Length of the longest waveform in the dataset.

    fs : int, optional
        The constant sampling frequency for the waveforms, if provided.

    Xtrain, Xtest : dict, optional
        Train and test subsets randomly split using `train_test_split`, if 
        required. These are views into `strains`, without associated labels.

    Notes
    -----
    - Unlike `Base`, this class does not track class labels.
    - Train/Test split is still supported but is not stratified.
    
    """
    def __init__(self,
                 strains_array: np.ndarray,
                 *,
                 fs: int,
                 strain_limits=None,
                 whitened=False,
                 random_seed=None):
        """Initialize an UnlabeledWaves dataset.

        This constructor processes a NumPy array of gravitational wave signals,
        storing them in a structured dictionary while optionally discarding
        unnecessary zero-padding. Unlike `Base`, this class does not support
        labeled categories nor requires metadata, but retains support for
        dataset splitting and signal management.

        Parameters
        ----------
        strains_array : np.ndarray
            A 2D array containing gravitational wave signals, where each row 
            represents a separate waveform, possibly zero-padded.

        fs : int
            The assumed constant sampling frequency for the waveforms.

        strain_limits : list[tuple[int, int]] | None, optional
            A list of (start, end) indices defining the valid range for each 
            waveform in `strains_array`. If None, waveforms are assumed to 
            contain no unnecessary padding.
        
        whitened : bool, optional
            If True, it is assumed that signals in `strains_array` have already
            been whitened. This effectively changes some of the behaviour of
            the class when treating data internally.

        random_seed : int, optional
            Seed used to initialize the random number generator (RNG), as well as
            for calling :func:`sklearn.model_selection.train_test_split` to
            generate the Train and Test subsets.

        Notes
        -----
        - A dummy class label ('unique': 1) is assigned for compatibility
          inside the `strains` dict.
        - Metadata is omitted in this class.
        - The dataset structure supports train/test splitting, but labels are 
          ignored.
        - TODO: Implement optional explicit time arrays as argument for time
          varying sampling (and all corresponding checks).
        
        """
        self.classes = {'unique': 1}  # Dummy class.
        self.strains = self._unpack_strains(strains_array, strain_limits)
        self._gen_labels()  # Dummy labels.
        self.fs = fs
        # self.metadata: pd.DataFrame = None  # OMMITED IN THIS CLASS
        
        # Number of nested layers in strains' dictionary. Keep updated always:
        self._dict_depth: int = dictools.get_depth(self.strains)

        self.max_length = self._find_max_length()
        self.random_seed = random_seed  # SKlearn train_test_split doesn't accept a Generator yet.
        self.rng = np.random.default_rng(self.random_seed)

        self.padding = {}

        # Whitening related attributes.
        self.whitened = whitened
        self.whiten_params = {}
        self.nonwhiten_strains = None if self.whitened else self.strains 

        # Time tracking related attributes.
        self._track_times = False  # If True, self.times must be not None.
        self.times: dict = None
        
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

    def _unpack_strains(self, strain_array: np.ndarray, strain_limits: np.ndarray = None) -> dict:
        num_signals = strain_array.shape[0]

        if strain_limits is None:
            extracted_signals = {i: strain_array[i, :] for i in range(num_signals)}
        elif strain_limits.shape == (2,):
            start, end = strain_limits
            extracted_signals = {i: strain_array[i, start:end] for i in range(num_signals)}
        elif strain_limits.shape == (num_signals, 2):
            extracted_signals = {i: strain_array[i, start:end] for i, (start, end) in enumerate(strain_limits)}
        else:
            raise ValueError("Invalid shape for strain_limits. Must be None, (2,), or (N,2).")

        # Add the outer (class) layer:
        class_name = next(iter(self.classes))  # Get the first class name

        return {class_name: extracted_signals}
    

class InjectedUnlabeledWaves(UnlabeledBaseMixin, BaseInjected):
    """Dataset class for injected gravitational wave signals without labels.

    This class extends `Base`, modifying its behavior to handle injections in
    `UnlabeledWaves` datasets, where gravitational wave signals are provided
    without associated labels.

    Attributes
    ----------
    TODO

    Notes
    -----
    - Unlike `BaseInjected`, this class does not track class labels.
    - Train/Test split is still supported but is not stratified.
    
    """
    def __init__(self,
                 clean_dataset: UnlabeledWaves,
                 psd: np.ndarray | Callable = None,
                 noise_length: int = None,
                 freq_cutoff: int | float = None,
                 freq_butter_order: int | float = None,
                 noise_instance: synthetic.NonwhiteGaussianNoise = None,
                 detector: str = '',
                 random_seed: int = None):
        """Initialize an InjectedUnlabeledWaves dataset.

        This constructor is built from a previous UnlabeledWaves instance.

        If train/test subsets are present, they too are updated when performing
        injections or changing units, but only through re-building them from
        the main 'strains' attribute using the already generated indices.
        Original train/test subsets from the clean dataset are not inherited.

        .. warning::
            Initializing this class does not perform the injections! For
            that use the method 'gen_injections'.

        Parameters
        ----------
        clean_dataset : UnlabeledWaves

        psd : np.ndarray | Callable, optional
            Power Spectral Density of the detector's sensitivity in the range
            of frequencies of interest. Can be given as a callable function
            whose argument is expected to be an array of frequencies, or as a
            2d-array with shape (2, psd_length) so that
            
            ```
            psd[0] = frequency_samples
            psd[1] = psd_samples
            ```.

            If not given, it will be assumed that the dataset lives in the
            whitened space.
            
            .. note::
                `psd` is also used to compute the 'asd' attribute, if given.
        
        noise_length : int, optional
            Length of the background noise array to be generated for later use.
            It should be at least longer than the longest signal expected to be
            injected.

        freq_cutoff : int | float, optional
            Frequency cutoff below which no noise bins will be generated in the
            frequency space, and also used for the high-pass filter applied to
            clean signals before injection.

        freq_butter_order : int | float, optional
            Butterworth filter order. For signals above 100 Hz it's usually
            enough with order 4 to 6.
            See (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
            for more information.

        noise_instance : NonwhiteGaussianNoise-like, optional
            [Experimental] Instead of generating random Gaussian noise, an
            already generated (or real) noise array can be given.

            .. warning::
                This option still needs to be properly integrated and tested.

        detector : str, optional
            GW detector name.
            Not used, just for identification.

        random_seed : int, optional
            Value passed to 'sklearn.model_selection.train_test_split' to
            generate the Train and Test subsets.
            Saved for reproducibility purposes, and also used to initialize
            Numpy's default RandomGenerator.

        Notes
        -----
        - A dummy class label ('unique': 1) is assigned for compatibility.
        - Metadata is omitted in this class.
        - The dataset structure supports train/test splitting, but labels are 
          not relevant.
        - This constructor is a reimplementation of `Base.__init__` adapted for
          a single (dummy) class.
        
        """
        if not clean_dataset.fs:
            raise ValueError("`fs` must be defined in order to perform injections")

        # Inherit clean strain instance attributes.
        #----------------------------------------------------------------------
        self.fs = clean_dataset.fs

        if clean_dataset.nonwhiten_strains is None:
            # Whitened space case (no access to strains before whitening).
            self._data_in_white_space = True
            self.strains_clean = deepcopy(clean_dataset.strains)
        else:
            # Non-whitened case (access to original strains).
            self._data_in_white_space = False
            self.strains_clean = deepcopy(clean_dataset.nonwhiten_strains)
        
        self.classes = clean_dataset.classes.copy()  # Dummy class.
        self.labels = self.labels = clean_dataset.labels.copy()  # Dummy labels.
        self._track_times = clean_dataset._track_times
        self.times = deepcopy(clean_dataset.times) if self._track_times else None
        self.padding = clean_dataset.padding.copy()
        self.max_length = clean_dataset.max_length

        # Noise instance and related attributes.
        #----------------------------------------------------------------------
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.detector = detector

        # Highpass parameters applied when generating the noise array.
        self.freq_cutoff = freq_cutoff
        self.freq_butter_order = freq_butter_order

        if self._data_in_white_space:
            self._psd, self.psd_array = None, None
            self._asd, self.asd_array = None, None
        else:
            self._psd, self.psd_array = self._setup_psd(psd)
            self._asd, self.asd_array = self._setup_asd_from_psd(psd)

        if noise_instance is None:
            # Generate synthetic non-white Guassian noise.
            if psd is None:
                raise ValueError(
                    "in order to generate synthetic background, 'psd' must be"
                    " provided."
                )
            self.noise = self._generate_background_noise(noise_length)
        else:
            # EXPERIMENTAL OPTION TO ALLOW THE USE OF REAL OR PRE-GENERATED
            # BACKGROUND NOISE.
            if not isinstance(noise_instance, synthetic.NonwhiteGaussianNoise):
                raise TypeError(
                    "'noise_instance' must be a valid noise type"
                    f" ({type(noise_instance)} was given)"
                )
            self.noise = noise_instance

        # Injection related:
        #----------------------------------------------------------------------
        self.strains = None
        self._dict_depth = clean_dataset._dict_depth + 1  # Depth of the strains dict.
        self.snr_list = []
        self.injection_snr_scales = None
        self.injections_per_snr = 1  # Default value.
        self.whitened = self._data_in_white_space
        self.whiten_params = None
        
        # Train/Test subset views:
        #----------------------------------------------------------------------
        if clean_dataset.Xtrain is not None:
            self.Xtrain = {k: None for k in clean_dataset.Xtrain.keys()}
            self.Xtest = {k: None for k in clean_dataset.Xtest.keys()}
            self.Ytrain = clean_dataset.Ytrain
            self.Ytest = clean_dataset.Ytest
            self.id_train = clean_dataset.id_train
            self.id_test = clean_dataset.id_test
        else:
            self.Xtrain = None
            self.Xtest = None
            self.Ytrain = None
            self.Ytest = None
            self.id_train = None
            self.id_test = None


class CoReWaves(Base):
    """Manage all operations needed to perform over a noiseless CoRe dataset.

    Initial strains and metadata are obtained from a CoReManager instance.

    NOTE: This class treats as different classes (categories) each equation of
    state (EOS) present in the CoReManager instance.

    NOTE^2: This class adds a time attribute with time samples related to each
    GW.

    Workflow:
    
    - Load the strains from a CoreWaEasy instance, discarding or cropping those
      indicated with their respective arguments.
    
    - Resample.
    
    - Project onto the ET detector arms.
    
    - Change units and scale from geometrized to IS and vice versa.
    
    - Export the (latest version of) dataset to a HDF5.
    
    - Export the (latest version of) dataset to a GWF.

    Attributes
    ----------
    classes : dict
        Dict of strings and their integer labels, one per class (category).
        The keys are the name of the Equation of State (EOS) used to describe
        the physics behind the simulation which produced each strain.
    
    strains : dict {class: {id: gw_strain} }
        Strains stored as a nested dictionary, with each strain in an
        independent array to provide more flexibility with data of a wide
        range of lengths.
        The class key is the name of the class, a string which must exist in
        the 'classes' list.
        The 'id' is an unique identifier for each strain, and must exist in the
        `self.metadata.index` column of the metadata DataFrame.
        Initially, an extra depth layer is defined to store the polarizations
        of the CoRe GW simulated data. After the projection this layer will be
        collapsed to a single strain.
    
    times : dict {class: {id: gw_time_points} }
        Time samples associated with the strains, following the same structure.
        Useful when the sampling frequency is variable or different between strains.
    
    metadata : pandas.DataFrame
        All parameters and data related to the strains.
        The order is the same as inside 'strains' if unrolled to a flat list
        of strains up to the second depth level (the id.).
        Example:
        
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
    
    units : str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.
    
    fs : int, optional
        Initially this attribute is None because the initial GW from CoRe are
        sampled at different and non-constant sampling frequencies. After the
        resampling, this attribute will be set to the new global sampling frequency.

        Caveat: If the 'times' attribute is present, this value is ignored.
        Otherwise it is assumed all strains are constantly sampled to this.

    """
    def __init__(self,
                 *,
                 coredb: ioo.CoReManager,
                 classes: dict[str],
                 discarded: set,
                 cropped: dict,
                 # Source:
                 distance: float,
                 inclination: float,
                 phi: float):
        """Initialize a CoReWaves dataset.

        TODO

        Parameters
        ----------
        coredb : ioo.CoReManager
            Instance of CoReManager with the actual data.
        
        classes : dict[str]
            Dictionary with the Equation of State (class) name as key and the
            corresponding label index as value.
        
        discarded : set[str]
            Set of GW IDs to discard from the dataset.
        
        cropped : dict[str]
            Dictionary with the class name as key and the corresponding
            cropping range as value. The range is given as a tuple of the form
            (start_index, stop_index).
        
        distance : float
            Distance to the source in Mpc.
        
        inclination : float
            Inclination of the source in radians.
        
        phi : float
            Azimuthal angle of the source in radians.

        """
        if not isinstance(coredb, ioo.CoReManager):
            raise TypeError("Expected 'coredb' to be an instance of CoReManager.")

        self._check_classes_dict(classes)
        self.classes = classes
        self.discarded = discarded
        self.cropped = cropped
        # Source parameters
        self.distance = distance
        self.inclination = inclination
        self.phi = phi

        self.units = 'IS'
        self.strains, self.times, self.metadata = self._get_strain_and_metadata(coredb)
        self._track_times = True
        self._dict_depth = dictools.get_depth(self.strains)
        self._gen_labels()
        self.max_length = self._find_max_length()

        self.fs = None  # Set up after resampling
        self.random_seed = None  # Set if calling the 'build_train_test_subsets' method.
        self.rng = np.random.default_rng(self.random_seed)

        self.padding = {}

        self.whitened = False
        self.whiten_params = {}
        self.nonwhiten_strains = self.strains

        # Train/Test subset splits (views into the same 'self.strains').
        #   Timeseries:
        self.Xtrain: np.ndarray = None
        self.Xtest: np.ndarray = None
        #   Labels:
        self.Ytrain: np.ndarray = None
        self.Ytest: np.ndarray = None
    
    def _get_strain_and_metadata(self, coredb: ioo.CoReManager) -> tuple[dict, dict, pd.DataFrame]:
        """Obtain the strain and metadata from a CoReManager instance.

        The strains are the Pluss and Cross polarizations obtained from the
        direct output of numerical relativistic simulations. They are expected
        to be projected at the detector afterwards, collapsing the polarization
        layer to a single strain per GW.
        
        Returns
        -------
        strains : dict{eos: {id: {pol: strain} } }
        
        times : dict{'eos': {'id': {pol: time_samples}} }
            Time samples associated to each GW.
            Since it has to follow the same nested structure as 'strains', but
            the time samples are the same among polarizations, for each GW both
            polarizations point to the same array in memory.
        
        metadata : pandas.DataFrame
            All parameters and data related to the strains.
            The order is the same as inside 'strains' if unrolled to a flat list
            of strains up to the second depth level (the id.).
        
        """
        strains = self._gen_empty_strains_dict()
        times = self._gen_empty_strains_dict()
        # Metadata columns/keys:
        index: list[str] = []
        mass: list[float] = []
        mass_ratio: list[float] = []
        eccentricity: list[float] = []
        lambda_tidal: list[float] = []  # Tidal deformability
        mass_starA: list[float] = []
        mass_starB: list[float] = []
        spin_starA: list[float] = []
        spin_starB: list[float] = []
        merger_pos: list[int] = []  # Index position of the merger inside the array.

        for eos in self.classes:
            # Get and filter out GW simulations.
            ids = set(coredb.filter_by('id_eos', eos).index)
            try:
                ids -= self.discarded[eos]
            except KeyError:
                pass  # No discards.
            ids = sorted(ids)  # IMPORTANT!!! Keep order to be able to trace back simulations.
            
            for id_ in ids:
                # CoRe Rh data (in IS units):
                times_, h_plus, h_cros = coredb.gen_strain(
                    id_, self.distance, self.inclination, self.phi
                )

                # Crop those indicated at the parameter file, and leave whole
                # the rest.
                try:
                    t0, t1 = self.cropped[eos][id_]
                except KeyError:
                    crop = slice(None)
                else:
                    crop = slice(
                        np.argmin(np.abs(times_-t0)),
                        np.argmin(np.abs(times_-t1))
                    )
                strains[eos][id_] = {
                    'plus': h_plus[crop],
                    'cross': h_cros[crop]
                }
                # Both polarizations have the same sampling times, hence we
                # point each time polarization to the same array in memory.
                times[eos][id_] = {}
                times[eos][id_]['plus'] = times[eos][id_]['cross'] = times_[crop]
                
                # The time is centered at the merger.
                i_merger = tat.find_time_origin(times_[crop])

                # Associated metadata:
                md = coredb.metadata.loc[id_]
                index.append(md['database_key'])
                mass.append(md['id_mass'])
                mass_ratio.append(md['id_mass_ratio'])
                eccentricity.append(md['id_eccentricity'])
                lambda_tidal.append(md['id_Lambda'])
                mass_starA.append(md['id_mass_starA'])
                mass_starB.append(md['id_mass_starB'])
                spin_starA.append(md['id_spin_starA'])
                spin_starB.append(md['id_spin_starB'])
                merger_pos.append(i_merger)
        
        metadata = pd.DataFrame(
            data=dict(
                mass=mass, mass_ratio=mass_ratio,
                eccentricity=eccentricity, lambda_tidal=lambda_tidal,
                mass_starA=mass_starA, mass_starB=mass_starB,
                spin_starA=spin_starA, spin_starB=spin_starB,
                merger_pos=merger_pos
            ),
            index=index
        )
        
        return strains, times, metadata
    
    def find_merger(self, strain: np.ndarray) -> int:
        return tat.find_merger(strain)

    def _update_merger_positions(self):
        """Update all 'merger_pos' tags inside the metadata attribute.
        
        Time arrays are defined with the origin at the merger. When the length
        of the strain arrays is modified, the index position of the merger
        must be updated.

        NOTE: This method updates ALL the merger positions.
        
        """
        for clas, id_ in self.keys(max_depth=2):
            times = self.times[clas][id_]
            # If more layers are present, only get the first instance of times
            # since all will be the same.
            if isinstance(times, dict):
                times = dictools.get_first_value(times)
            self.metadata.at[id_,'merger_pos'] = tat.find_time_origin(times)
    
    def resample(self, fs, verbose=False) -> None:
        """Resample strain and time arrays to a constant rate.

        Resample CoRe strains (from NR simulations) to a constant rate.

        This method updates the sampling frequency, the max_length and the
        merger_pos inside the metadata attribute.

        Parameters
        ----------
        fs : int
            The new sampling frequency in Hz.

        verbose : bool
            If True, print information about the resampling.
        
        """
        super().resample(fs, verbose)

        # Update side-effect attributes.
        self._update_merger_positions()
        if self.Xtrain is not None:
            self._update_train_test_subsets()

    def project(self, *, detector: str, ra: float, dec: float, geo_time: float, psi: float):
        """Project strains into the chosen detector at specified coordinates.

        Project strains into the chosen detector at specified coordinates,
        using Bilby.

        This collapses the polarization layer in 'strains' and 'times' to a
        single strain.
        The times are rebuilt taking as a reference point the merger (t = 0).
        
        Parameters
        ----------
        detector : str
            Name of the ET arm in Bilby for InterferometerList().
        
        ra, dec : float
            Sky position in equatorial coordinates.
        
        geo_time : int | float
            Time of injection in GPS.
        
        psi : float
            Polarization angle.

        Caveats
        -------
        - The detector's name must exist in Bilby's InterferometerList().
        - Only one arm can be chosen.
        
        """
        project_pars = dict(ra=ra, dec=dec, geocent_time=geo_time, psi=psi)
        for clas, id_ in self.keys(max_depth=2):
            hp = self.strains[clas][id_]['plus']
            hc = self.strains[clas][id_]['cross']
            
            # Drop the polarization layer.
            strain = detectors.project(
                hp, hc, parameters=project_pars, fs=self.fs, 
                nfft=2*self.fs, detector=detector
            )
            self.strains[clas][id_] = strain
            
            # Regenerate the time array with the merger located at the origin.
            duration = len(strain) / self.fs
            t_merger = self.find_merger(strain) / self.fs
            t0 = -t_merger
            t1 = duration - t_merger
            self.times[clas][id_] = tat.gen_time_array(t0, t1, self.fs)
        
        # Update side-effect attributes
        self._dict_depth = dictools.get_depth(self.strains)
        self._update_merger_positions()
        self.max_length = self._find_max_length()
        if self.Xtrain is not None:
            self._update_train_test_subsets()
    
    def shrink_to_merger(self, offset: int = 0) -> None:
        """Shrink strains and time arrays w.r.t. the merger.

        Shrink strains (and their associated time arrays) discarding the left
        side of the merger (inspiral), with a given offset in samples.

        This also updates the metadata column 'merger_pos'.

        .. warning::
            This is an irreversible action.

        Parameters
        ----------
        offset : int
            Offset in samples, relative to the merger position.

        """
        # Compute the equivalent shrinkage padding to apply for each signal at
        # the ID layer.
        padding = {}
        for id in self.metadata.index:
            i_merger = self.metadata.at[id, 'merger_pos']
            # Same shrinking limits for all possible strains below ID layer.
            padding[id] = (i_merger+offset, 0)
        
        self.shrink_strains(padding)

        # Update side-effect attributes.
        self._update_merger_positions()
        if self.Xtrain is not None:
            self._update_train_test_subsets()

    def convert_to_IS_units(self) -> None:
        """Convert data from scaled geometrized units to IS units.

        Convert strains and times from geometrized units (scaled to the mass
        of the system and the source distance) to IS units.
        
        Will raise an error if the data is already in IS units.
        
        """
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        for keys in self.keys():
            id_ = keys[1]
            mass = self.metadata.at[id_,'mass']
            strain = self.get_strain(*keys)
            times = self.get_times(*keys)

            strain *=  mass * MSUN_MET / (self.distance * MPC_MET)
            times *= mass * MSUN_SEC

        self.units = 'IS'

        # Update side-effect attributes.
        if self.Xtrain is not None:
            self._update_train_test_subsets()
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert data from IS to scaled geometrized units.
        
        Convert strains and times from IS to geometrized units, and scaled to the mass
        of the system and the source distance.

        Will raise an error if the data is already in geometrized units.
        
        """
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")
        
        for keys in self.keys():
            id_ = keys[1]
            mass = self.metadata.at[id_,'mass']
            strain = self.get_strain(*keys)
            times = self.get_times(*keys)
            
            strain /=  mass * MSUN_MET / (self.distance * MPC_MET)
            times /= mass * MSUN_SEC

        self.units = 'geometrized'

        # Update side-effect attributes.
        if self.Xtrain:
            self._update_train_test_subsets()


class InjectedCoReWaves(BaseInjected):
    """Manage injections of GW data from CoRe dataset.
    
    - Tracks index position of the merger.
    
    - Computes the SNR only at the ring-down starting from the merger.
    
    - Computes also the usual SNR over the whole signal and stores it for
      later reference (attr. 'whole_snr_list').

    Attributes
    ----------
    snr_list : list
        Partial SNR values at which each signal is injected.
        This SNR is computed ONLY over the Ring-Down section of the waveform
        starting from the merger, hence the name 'partial SNR'.

    whole_snr : dict
        Nested dictionary storing for each injection the equivalent SNR value
        computed over the whole signal, hence the name 'whole SNR'.
        Structure: {id_: {partial_snr: whole_snr}}

    TODO

    """
    def __init__(self,
                 clean_dataset: Base,
                 *,
                 psd: np.ndarray | Callable,
                 detector: str,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 random_seed: int):
        """
        Initializes an instance of the InjectedCoReWaves class.

        Parameters
        ----------
        clean_dataset : Base
            An instance of a BaseDataset class with noiseless signals.
        
        psd : np.ndarray | Callable
            Power Spectral Density of the detector's sensitivity in the
            range of frequencies of interest.
            Can be given as a callable function whose argument is
            expected to be an array of frequencies, or as a 2d-array
            with shape (2, psd_length) so that

            ```
                psd[0] = frequency_samples
                psd[1] = psd_samples.
            ```
            
            NOTE: It is also used to compute the 'asd' attribute (ASD).
        
        detector : str
            GW detector name.
        
        noise_length : int
            Length of the background noise array to be generated for
            later use.
            It should be at least longer than the longest signal
            expected to be injected.
        
        freq_cutoff : int | float
            Frequency cutoff for the filter applied to the signal.
        
        freq_butter_order : int | float
            Order of the Butterworth filter applied to the signal.
        
        random_seed : int
            Random seed for generating random numbers.
        
        """
        super().__init__(
            clean_dataset,
            psd=psd,
            detector=detector,
            noise_length=noise_length,
            freq_cutoff=freq_cutoff,
            freq_butter_order=freq_butter_order,
            random_seed=random_seed
        )

        self.whole_snr = {id_: {} for id_ in self.labels}

    def _update_merger_positions(self):
        """Update all 'merger_pos' tags inside the metadata attribute.
        
        Time arrays are defined with the origin at the merger. When the length
        of the strain arrays is modified, the index position of the merger
        must be updated.

        NOTE: This method updates ALL the merger positions.
        
        """
        for clas, id_ in self.keys(max_depth=2):
            # Same time array for all SNR variations.
            times = dictools.get_first_value(self.times[clas][id_])
            self.metadata.at[id_,'merger_pos'] = tat.find_time_origin(times)
    
    def gen_injections(self,
                       snr: int|float|list,
                       snr_offset: int = 0,
                       randomize_noise: bool = False,
                       random_seed: int = None,
                       injections_per_snr: int = 1,
                       verbose=False):
        """Inject all strains in simulated noise with the given SNR values.

        See 'BaseInjected.gen_injections' for more details.
        
        Parameters
        ----------
        snr : int | float | list

        snr_offset : int
            An offset (relative to the position of the merger) added to the
            start of the segment of the clean signal used for SNR calculation.
            If the SNR computation needs to include a portion of signal BEFORE
            the merger, the offset should be negative.

        randomize_noise : bool
            If True, the noise segment is randomly chosen before the injection.
            This can be used to avoid having the same noise injected for all
            clean strains.
            False by default.
            
            NOTE: To avoid the possibility of repeating the same noise section
            in different injections, the noise realization must be reasonably
            large, e.g:
                
                `noise_length > n_clean_strains * self.max_length * len(snr)`
        
        random_seed : int, optional
            Random seed for the noise realization.
            Only used when randomize_noise is True.

        injections_per_snr : int
            Number of injections per SNR value.
            1 by default.
        
        Notes
        -----
        - If whitening is intended to be applied afterwards it is useful to
          pad the signal in order to avoid the window vignetting produced by
          the whitening itself. This pad will be cropped afterwards.
        
        - New injections are stored in the 'strains' atrribute, with the pad
          associated to all the injections performed at once. Even when
          whitening is also performed right after the injections.
        
        Raises
        ------
        ValueError
            Once injections have been performed at a certain SNR value, there
            cannot be injected again at the same value. Trying it will trigger
            this exception.
        
        """
        super().gen_injections(
            snr, snr_offset=snr_offset, randomize_noise=randomize_noise,
            random_seed=random_seed, injections_per_snr=injections_per_snr,
            verbose=verbose
        )

        self._update_merger_positions()
    
    def _inject(self,
                strain: np.ndarray,
                snr: int | float,
                *,
                id: str,
                snr_offset: int | list,
                pos: int = 0) -> np.ndarray:
        """Inject a strain at 'snr' into noise using 'self.noise' instance.

        Parameters
        ----------
        strain : NDArray
            Signal to be injected into noise.
        
        snr : int | float
            Targeted signal-to-noise ratio.
        
        id : str
            Signal identifier (2nd layer of 'strains' dict).
        
        snr_offset : int
            An offset (relative to the position of the merger) added to the
            start of the segment of the clean signal used for SNR calculation.

        pos : int, optional
            Index position in the noise array where to inject the signal. 0 by
            default.
        
        Returns
        -------
        injected : NDArray
            Injected signal.

        scale : float
            Scale factor applied to the signal.
        
        NOTES
        -----
        - The SNR is computed over the Post-Merger only.

        - The metadata is expected to reflect the original state of the strains
          previous to any padding performed right before calling this function,
          which may be done to avoid the vignette effect.
        
        """
        clas = self.find_class(id)
        merger_pos = self.metadata.at[id,'merger_pos']
        original_length = len(self.strains_clean[clas][id])

        i0 = merger_pos + snr_offset
        i1 = (original_length - merger_pos) + snr_offset
        injected, scale = self.noise.inject(strain, snr=snr, snr_lim=(i0, i1), pos=pos)

        # Compute the equivalent SNR over the entire waveform.
        self.whole_snr[id][snr] = self.noise.snr(strain*scale)

        return injected, scale
    
    def whiten(self,
               *,
               flength,
               highpass=None,
               normed=False,
               shrink=0,
               window='hann',
               verbose=False):
        """Whiten injected strains.
        
        Calling this method performs the whitening of all injected strains.
        Strains are later cut to their original size before adding the pad,
        to remove the vigneting.
        
        .. warning::
            This is an irreversible action; if the original injections need
            to be preserved it is advised to make a copy of the instance before
            performing the whitening.
        
        """
        super().whiten(
            flength=flength,
            highpass=highpass,
            normed=normed,
            shrink=shrink,
            window=window,
            verbose=verbose
        )

        self._update_merger_positions()


def _check_vectorized(func):
    """Check if a function is vectorized (i.e., supports NumPy arrays element-wise)."""
    test_input = np.array([1, 2, 3])  # Small test array
    
    # Run the function; any raised error propagates immediately
    output = func(test_input)

    # Ensure the output is a NumPy array of the same shape
    if not (isinstance(output, np.ndarray) and (output.shape == test_input.shape)):
        raise TypeError("The provided function is not properly vectorized. Use numpy.vectorize if needed.")
