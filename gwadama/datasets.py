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


__all__ = ['CoReDataset', 'InjectedDataset', 'SyntheticDataset']


class BaseDataset:
    """Base class for all datasets.
    
    ATTRIBUTES
    ----------
    classes: list[str]
        List of labels, one per class (category).

    strains: dict {class_i: gw_strains}
        Latest version of the strains.
    
    strains_history: dict {"step": strains_after_that_step}
        Copy of 'strains' at each step of the process.
    
    units: str
        Flag indicating whether the data is in 'geometrized' or 'IS' units.
    
    """
    def __init__(self):
        """Overwrite when inheriting!"""

        # Need to be defined:
        self.classes: list[str] = None
        self.strains = None
        self.metadata = None
        self.units = None
        self.distance = None  # Mpc

        # Additional attributes used by the methods of this class:
        self.strains_history = {'initial': deepcopy(self.strains)}
        self.sample_rate = None
        self.max_length = self._find_max_length()

    def resample(self, sample_rate, verbose=False) -> None:
        for clas, key in self.unroll_strain_indices():
            strain = self.strains[clas][key]
            strain_resampled, sf_up, factor_down = resample(strain, sample_rate)
            self.strains[clas][key] = strain_resampled
            if verbose:
                print(
                    f"Strain {clas}::{key} up. to {sf_up} Hz, down by factor {factor_down}"
                )
        self.strains_history[f'resampling-{sample_rate}'] = deepcopy(self.strains)
        self.sample_rate = sample_rate
    
    def export_strains_to_hdf5(self, file: str) -> None:
        """Export all strains and their metadata to a single HDF5 file."""

        ioo.save_to_hdf5(file, data=self.strains, metadata=self.metadata)

    def unroll_strain_indices(self) -> list:
        """Return the unrolled combinations of all strains.

        Return the unrolled combinations of all strains by a hierarchical
        recursive search. Useful to iterate over all the strains.
        
        E.g: A strain dataset with a two-level hierarchy, class-key, can be
        iterated over doing:
        ```
        for clas, key in self._unroll_strain_indices():
            do_stuff(clas, key)
        ```
        instead of:
        ```
        for clas in self.classes:
            for key in self.strains[clas].keys():
                do_stuff(clas, key)
        ```
        
        RETURNS
        -------
        unrolled: list
            The unrolled combination in a Python list.
        
        """
        unrolled = unroll_nested_dictionary_keys(self.strains)

        return unrolled
    
    def get_strain(self, *indices) -> np.ndarray:
        """Get a single strain from the complete index coordinates.
        
        This is just a shortcut to avoid having to write several squared
        brackets.
        
        """
        strain = self.strains
        for key in indices:
            strain = strain[key]
        
        return strain


    def _find_max_length(self) -> int:
        """Return the length of the longest signal present in strains."""

        max_length = 0
        for clas, key in self.unroll_strain_indices():
            l = self.strains[clas][key].shape[-1]
            if l > max_length:
                max_length = l

        return max_length


class SyntheticDataset(BaseDataset):
    """Class for building and managing synthetically generated wavforms.

    Workflow:  # TODO: Update when finished the refactor.
    1- Generate (random sample) all metadata, i.e. the waveforms'
        parameters:
            `Dataset.gen_metadata()`
    2- Generate the actual dataset, i.e. waveforms, as time series:
            `Dataset.gen_dataset()`
    3- Split the dataset into train / test subsets:
            `Dataset.train_test_split(train_size)
    4- Generate the background noise to make injections.

    NOTES
    -----
    - Times begin at 0; adjust 'peak_time' w.r.t. this choice.
    - Assumes balanced classes.
    - Numpy's random state of the RNG is saved before each step in which is
      called (e.g. metadata generation).

    """
    classes = {'SG': 'Sine Gaussian', 'G': 'Gaussian', 'RD': 'Ring-Down'}
    n_classes = len(classes)

    def __init__(self, *, n_samples, length, peak_time, amp_threshold, tukey_alpha,
                 sample_rate, wave_parameters_limits, train_size, noise_psd, noise_duration,
                 random_seed=None):
        self.n_samples = n_samples
        self.n_samples_total = self.n_samples * self.n_classes
        self.length = length
        self.sample_rate = sample_rate
        self.wave_limits = wave_parameters_limits
        self.peak_time = peak_time  # Reffered to as 't0' at waveform functions.
        self.tukey_alpha = tukey_alpha
        self.amp_threshold = amp_threshold
        self.train_size = train_size
        self.random_seed = random_seed
        self.random_states = {}
        self.rng = np.random.default_rng(random_seed)

        self.metadata = None
        self.dataset = None  # (samples, features)
        self.times = None

        # ORIGINAL (NOISE-FREE) WAVEFORMS
        # Timeseries data:
        self.Xtrain = None  # (samples, features)
        self.Xtest = None   # (samples, features)
        # Labels:
        self.Ytrain = None
        self.Ytest = None
        # Indices w.r.t. 'self.dataset':
        self.Itrain = None
        self.Itest = None

        # BACKGROUND NOISE FOR INJECTIONS
        self.noise = NonwhiteGaussianNoise(
            duration=noise_duration, psd=noise_psd, sample_rate=self.sample_rate, rng=self.rng
        )

    def gen_metadata(self):
        self.random_states['gen_metadata'] = self.rng.bit_generator.state
        md = pd.DataFrame(
            np.zeros((self.n_samples_total, 5), dtype=float),
            columns=('Class', 'f0', 'Q', 'hrss', 'duration')
        )
        for iclass, class_ in enumerate(self.classes):
            for i in range(iclass*self.n_samples, (iclass+1)*self.n_samples):
                f0, Q, hrss, duration = self._gen_parameters[class_](self)
                md.at[i, 'Class'] = class_
                md.at[i, 'f0'] = f0
                md.at[i, 'Q'] = Q
                md.at[i, 'hrss'] = hrss
                md.at[i, 'duration'] = duration  # Will be adjusted afterwards to take into account
                                                 # the amplitude threshold.

        self.metadata = md

    def _gen_parameters_sine_gaussian(self):
        """Generate random parameters for a single Sine Gaussian."""

        lims = self.wave_limits
        thres = self.amp_threshold
        f0   = self.rng.integers(lims['mf0'], lims['Mf0'])  # Central frequency
        Q    = self.rng.integers(lims['mQ'], lims['MQ']+1)  # Quality factor
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = 2 * Q / (np.pi * f0) * np.sqrt(-np.log(thres))
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_gaussian(self):
        """Generate random parameters for a single Gaussian."""

        lims = self.wave_limits
        f0   = None  #  Casted to np.nan afterwards.
        Q    = None  #-/
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = self.rng.uniform(lims['mT'], lims['MT'])  # Duration
        
        return (f0, Q, hrss, duration)

    def _gen_parameters_ring_down(self):
        """Generate random parameters for a single Ring-Down."""

        lims = self.wave_limits
        thres = self.amp_threshold
        f0   = self.rng.integers(lims['mf0'], lims['Mf0'])  # Central frequency
        Q    = self.rng.integers(lims['mQ'], lims['MQ']+1)  # Quality factor
        hrss = self.rng.uniform(lims['mhrss'], lims['Mhrss'])
        duration = -np.sqrt(2) * Q / (np.pi * f0) * np.log(thres)
        
        return (f0, Q, hrss, duration)
    
    wave_functions = {
        'SG': sine_gaussian_waveform,
        'G': gaussian_waveform,
        'RD': ring_down_waveform
    }

    _gen_parameters = {
        'SG': _gen_parameters_sine_gaussian,
        'G': _gen_parameters_gaussian,
        'RD': _gen_parameters_ring_down
    }

    def gen_dataset(self):
        if self.metadata is None:
            raise AttributeError("'metadata' needs to be generated first!")

        self.times = np.arange(0, self.length/self.sample_rate, 1/self.sample_rate)
        self.dataset = np.empty((self.n_samples_total, self.length))
        
        for i in range(self.n_samples_total):
            params = self.metadata.loc[i].to_dict()
            class_ = params['Class']
            self.dataset[i] = self.wave_functions[class_](
                self, self.times, t0=self.peak_time, **params
            )

        self._apply_threshold_windowing()

    def _apply_threshold_windowing(self):
        """Shrink waves in the dataset and update its duration in the metadata.

        Shrink them according to their pre-computed duration in the metadata to
        avoid almost-but-not-zero edges, and correct those marginal durations
        longer than the window.

        """
        for i in range(self.n_samples_total):
            duration = self.metadata.at[i,'duration']
            ref_length = int(duration * self.sample_rate)
            
            if self.metadata.at[i,'Class'] == 'RD':
                # Ring-Down waves begin at the center.
                i0 = self.length // 2
                i1 = i0 + ref_length
            else:
                # SG and G are both centered.
                i0 = (self.length - ref_length) // 2
                i1 = self.length - i0

            new_lenght = i1 - i0
            if i0 < 0:
                new_lenght += i0
                i0 = 0
            if i1 > self.length:
                new_lenght -= i1 - self.length
                i1 = self.length

            window = sp.signal.windows.tukey(new_lenght, alpha=self.tukey_alpha)
            self.dataset[i,:i0] = 0
            self.dataset[i,i0:i1] *= window
            self.dataset[i,i1:] = 0

            self.metadata.at[i,'duration'] = new_lenght / self.sample_rate

    def train_test_split(self):
        labels = np.repeat(range(self.n_classes), self.n_samples)
        indices = range(self.n_samples_total)  # keep track of samples after shuffle.
        
        # WARNING: This function returns A COPY, so if I modify any of the
        # original data (e.g. a change in units) I will need to either apply
        # the exact same transformation to the splitted data or compute
        # again the split, making sure the random distribution is equal or
        # at least tracked!!!
        Xtrain, Xtest, Ytrain, Ytest, Itrain, Itest = train_test_split(
            self.dataset, labels, indices, train_size=self.train_size,
            random_state=self.random_seed, shuffle=True, stratify=labels
        )

        # Timeseries data:
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        # Labels:
        self.Ytrain = Ytrain
        self.Ytest = Ytest
        # Indices:
        self.Itrain = Itrain
        self.Itest = Itest

    def setup_all(self):
        self.gen_metadata()
        self.gen_dataset()
        self.train_test_split()

    def inject_train_test(self, *, snr, noise_pos, highpass_orders=None):
        """Inject training and testing sets into the pre-computed noise.

        Inject training and testing sets into the pre-computed noise at the
        given index position w.r.t. the noise array, at the specified SNR.
        Optionally performs a highpass filter to remove lowest frequencies.

        highpass_orders: list, optional
            Orders [N, wn] of the Butterworth filter applied to perform the
            highpass.

        """
        Xtrain = self._inject_signals(self.Xtrain, snr=snr, position=noise_pos)
        Xtest = self._inject_signals(self.Xtest, snr=snr, position=noise_pos)

        if highpass_orders is not None:
            N, wn = highpass_orders
            Xtrain = self._highpass_signals(Xtrain, N=N, wn=wn)
            Xtest = self._highpass_signals(Xtest, N=N, wn=wn)

        return Xtrain, Xtest

    def _inject_signals(self, signals, *, snr, position):
        injections = np.empty_like(signals)
        n = signals.shape[0]
        for i in range(n):
            injections[i], _ = self.noise.inject(
                signals[i], snr=snr, sample_rate=self.sample_rate, pos=position, l2_normed=False
            )

        return injections

    def _highpass_signals(self, signals, *, N, wn):
        filtered = np.empty_like(signals)
        n = signals.shape[0]
        for i in range(n):
            filtered[i] = self.noise.apply_frequency_filter(
                signals[i], N=N, wn=wn, btype='highpass'
            )

        return filtered        


class CoReDataset(BaseDataset):
    """Manage all operations needed to perform over a noiseless CoRe dataset.

    Initial strains and metadata are expected to be obtained from a CoReManager
    instance.

    NOTE: By default this class treats as different classes (categories) each
    equation of state (EOS) present in the CoReManager instance.

    Workflow:  # TODO: Update this shit
    - Load the strains from a CoreWaEasy instance, discarding or cropping those
      indicated with their respective arguments.
    - Resample.
    - Project onto the ET detector arms.
    - Change units and scale from geometrized to IS and vice versa.
    - Export the (latest version of) dataset to a HDF5.
    - Export the (latest version of) dataset to a GWF.

    ATTRIBUTES
    ----------
    strains: dict {eos_i: gw_strains}
        Latest version of the strains.
    
    strains_history: dict {"step": strains_after_that_step}
        Copy of 'strains' at each step of the process.
    
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
        self.strains_history = {'initial': deepcopy(self.strains)}
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

    def _init_strains_dict(self) -> dict:
        return {clas: {} for clas in self.classes}
    
    def project(self, *, detector: str, ra, dec, geo_time, psi):
        """Project strains into the ET detector at specified coordinates.
        
        PARAMETERS
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
        for clas, key in self.unroll_strain_indices():
            times, hp, hc = self.strains[clas][key]
            projected = project_et(
                hp, hc, parameters=project_pars, sf=self.sample_rate, 
                nfft=2*self.sample_rate, detector=detector
            )
            self.strains[clas][key] = np.stack([times, projected])
        self.strains_history['projected'] = deepcopy(self.strains)
    
    def convert_to_IS_units(self) -> None:
        """Convert strains from geometrized to IS units."""
        
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        for clas, key in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'IS'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert strains from IS to geometrized units."""
        
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")

        for clas, key in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'geometrized'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.

    def export_strains_to_hdf5(self, file: str):
        """Save strains and their metadata to a HDF5 file.
        
        CAUTION! THIS DOES NOT SAVE THE ENTIRE INSTANCE!
        
        """
        ioo.save_to_hdf5(file=file, data=self.strains, metadata=self.metadata)


class InjectedDataset(BaseDataset):
    """Manage an injected dataset with multiple SNR values.

    - Single noise realization, the same for all injections.
    - Manage multiple SNR values.
    - Export all strains to individual GWFs.
    - Unit conversion is performed on both strains and clean strains. However
      the noise instance is always in IS units.

    ATTRIBUTES
    ----------
    strains: dict {eos_i: dict {snr: NDArray} }
        Strains injected in noise. Each GW is the responses of the E1 detector
        times the number of SNR into which GW have been injected.
    
    strains_clean: dict {eos_i: gw_strains}
        Clean strains used for injection.
    
    psd: NDArray | function
        Chosen PSD of the simulated detector. If NDArray, it must have shape
        (2, psd_length) so that:
            psd[0] = time samples
            psd[1] = psd samples
    
    detector: str
        Name of the simulated detector.

    """
    def __init__(self,
                 clean_dataset: BaseDataset, *,
                 psd: np.ndarray | Callable,
                 detector: str,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 rng: np.random.Generator | int):
        """Initialize the injected dataset from a clean dataset.

        Relevant attributes are inherited from the clean dataset instance,
        which can be any inherited from BaseDataset whose strains have not
        been injected yet.
        
        WARNING: This method does not perform the injections! For that use the
        method 'gen_injections'.
        
        """
        # Inherit clean strain instance attributes.
        self.classes = clean_dataset.classes
        self.sample_rate = clean_dataset.sample_rate
        self.strains_clean = deepcopy(clean_dataset.strains)
        self.metadata = deepcopy(clean_dataset.metadata)
        self.max_length = clean_dataset.max_length
        self.units = clean_dataset.units
        self.distance = clean_dataset.distance

        # Noise instance and related attributes.
        self.psd, self._psd = self._setup_psd(psd)
        self.detector = detector
        self.freq_cutoff = freq_cutoff
        self.freq_butter_order = freq_butter_order
        self.rng = np.random.default_rng(rng)
        self.noise = self._generate_background_noise(noise_length)

        # Injection related.
        self.strains = None
        self.snr_list = []
        self.pad = {}  # {snr: pad}
    
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
        
        PARAMETERS
        ----------
        snr: int | float | list
        
        pad: int, optional
            If given, add additional noise samples at both extremes of all
            strains. This is useful for applying whitening afterwards, to avoid
            vignetting at the edges of the injected signal.
            The new length for each strain will be:
                `len(injected) == len(old_strain) + 2*pad`
        
        """
        if isinstance(snr, (int, float)):
            snr_list = list(snr)
        elif isinstance(snr, list):
            snr_list = snr
        else:
            raise TypeError(f"'{type(snr)}' is not a valid 'snr' type")
        
        if self.strains is None:
            self.strains = self._init_strains_dict()
        
        if self.units == 'geometrized':
            self.convert_to_IS_units()
            convert_back = True
        else:
            convert_back = False
        
        sr = self.sample_rate
        for clas, key in unroll_nested_dictionary_keys(self.strains_clean):
            gw_clean = self.strains_clean[clas][key]
            strain_clean_padded = np.pad(gw_clean[1], pad)
            t0 = gw_clean[0,0] - pad/sr
            t1 = gw_clean[0,-1] + pad/sr
            new_times = np.linspace(t0, t1, len(strain_clean_padded))
            
            for snr_ in snr_list:
                # Highpass filter to the clean signal. The noise realization is
                # already generated without frequency components lower than
                # the cutoff.
                strain_clean_padded = self.noise.highpass_filter(
                    strain_clean_padded, f_cut=self.freq_cutoff, f_order=self.freq_butter_order
                )
                injected, _ = self.noise.inject(strain_clean_padded, snr=snr_)
                self.strains[clas][key][snr_] = np.stack([new_times, injected])
        
        # Record new SNR values and related padding.
        self.snr_list += snr_list
        for snr_ in snr_list:
            self.pad[snr_] = pad
        
        if convert_back:
            self.convert_to_scaled_geometrized_units()
    
    def convert_to_IS_units(self) -> None:
        """Convert all strains from geometrized to IS units.
        
        Will raise an error if no injections are yet generated.
        
        """
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        self._convert_strain_clean_to_IS_units()
        self._convert_strain_to_IS_units()

        self.units = 'IS'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert all strains from IS to geometrized units.
        
        Will raise an error if no injections are yet generated.
        
        """
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")
        
        self._convert_strain_clean_to_scaled_geometrized_units()
        self._convert_strain_to_scaled_geometrized_units()

        self.units = 'geometrized'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.
    
    def _convert_strain_to_IS_units(self) -> None:
        for clas, key, snr in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][snr][0] *= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][snr][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_clean_to_IS_units(self) -> None:
        for clas, key in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains_clean[clas][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains_clean[clas][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_to_scaled_geometrized_units(self) -> None:
        for clas, key, snr in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains[clas][key][snr][0] /= mass * MSUN_SEC
            # Strain
            self.strains[clas][key][snr][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_clean_to_scaled_geometrized_units(self) -> None:
        for clas, key in self.unroll_strain_indices():
            mass = self.metadata[clas][key]['mass']
            # Time
            self.strains_clean[clas][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains_clean[clas][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def export_strains_to_gwf(self,
                              path: str,
                              channel: str,  # Name of the channel in which to save strains in the GWFs.
                              t0_gps: float = 0,
                              verbose=False) -> None:
        """Export all strains to GWF format, one file per strain."""

        for indices in self.unroll_strain_indices():
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


def unroll_nested_dictionary_keys(dictionary: dict) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    Useful to iterate over all keys of a nested dictionary without having to
    use multiple loops.

    PARAMETERS
    ----------
    dictionary: dict
        Nested dictionary.
    
    RETURNS
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
