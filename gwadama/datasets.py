from copy import deepcopy
from pathlib import Path
from typing import Callable

from corewaeasy import CoReManager
from gwpy.timeseries import TimeSeries
import numpy as np
from scipy.interpolate import make_interp_spline as sp_make_interp_spline

from . import ioo
from .detectors import project_et
from .synthetic import NonwhiteGaussianNoise
from .tima import resample
from .units import *


class BaseDataset:
    def __init__(self):
        """Overwrite when inheriting!"""

        # Need to be defined:
        self.eos: list[str] = None
        self.strains = None
        self.metadata = None
        self.units = None
        self.distance = None  # Mpc

        # Extra attributes present in this class:
        self.strains_history = {'initial': deepcopy(self.strains)}
        self.sample_rate = None
        self.max_length = self._find_max_length()

    def resample(self, sample_rate, verbose=False):
        for eos, key in self.unroll_strain_indices():
            strain = self.strains[eos][key]
            strain_resampled, sf_up, factor_down = resample(strain, sample_rate)
            self.strains[eos][key] = strain_resampled
            if verbose:
                print(
                    f"Strain {eos}::{key} up. to {sf_up} Hz, down by factor {factor_down}"
                )
        self.strains_history[f'resampling-{sample_rate}'] = deepcopy(self.strains)
        self.sample_rate = sample_rate
    
    def export_strains_to_hdf5(self, file: str):
        """CAUTION! THIS DOES NOT SAVE THE ENTIRE INSTANCE!"""

        ioo.save_to_hdf5(file, data=self.strains, metadata=self.metadata)
    
    def export_strains_to_gwf(self, path: str, t0_gps: float = 0, verbose=False):
        """Export all strains to GWF format.
        
        CAUTION! THIS DOES NOT SAVE THE ENTIRE INSTANCE!
        
        """
        for indices in self.unroll_strain_indices():
            times, strain = self.get_strain(*indices)
            ts = TimeSeries(
                data=strain,
                times=t0_gps + times,
                channel='ET:ET1'
            )
            indices = [str(i) for i in indices]
            file = Path(path) / ('strain_' + '_'.join(indices) + '.gwf')
            ts.write(file)

            if verbose:
                print("Strain exported to", file)


    def unroll_strain_indices(self) -> list:
        """Return the unrolled combinations of all strains.

        Return the unrolled combinations of all strains by a hierarchical
        recursive search. Useful to iterate over all the strains.
        
        E.g: A strain dataset with a two-level hierarchy, EOS-key, can be
        iterated over doing:
        ```
        for eos, key in self._unroll_strain_indices():
            do_stuff(eos, key)
        ```
        instead of:
        ```
        for eos in self.eos:
            for key in self.strains[eos].keys():
                do_stuff(eos, key)
        ```
        
        RETURNS
        -------
        unrolled: list
            The unrolled combination in a Python list.
        
        """
        unrolled = unroll_nested_dictionary_keys(self.strains)

        return unrolled
    
    def get_strain(self, *indices) -> np.ndarray:
        strain = self.strains
        for key in indices:
            strain = strain[key]
        
        return strain


    def _find_max_length(self) -> int:
        """Return the length of the longest signal present in strains."""

        max_length = 0
        for eos, key in self.unroll_strain_indices():
            l = self.strains[eos][key].shape[-1]
            if l > max_length:
                max_length = l

        return max_length


class CleanDataset(BaseDataset):
    """Manage all operations needed to perform over a noiseless dataset.

    - Load the strains from a CoreWaEasy instance, discarding or cropping those
      indicated at the parameter file.
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
                 cdb: CoReManager,
                 *,
                 eos: list[str],
                 discarded: set,
                 cropped: dict,
                 # Source:
                 distance: float, inclination: float, phi: float):
        self.eos = eos
        self.discarded = discarded
        self.cropped = cropped
        # Source parameters
        self.distance = distance
        self.inclination = inclination
        self.phi = phi

        # Note: I'm passing 'self.units' as return value just for consistence,
        # it could've been set its initial value ('IS') here.
        self.strains, self.metadata, self.units = self._get_strain_and_metadata(cdb)
        self.strains_history = {'initial': deepcopy(self.strains)}
        self.max_length = self._find_max_length()

        self.sample_rate = None  # Set up after resampling
    
    def _get_strain_and_metadata(self, cdb: CoReManager) -> dict[dict[np.ndarray]]:
        strains = self._init_strains_dict()
        metadata = self._init_strains_dict()
        units = 'IS'  # CoReManager.gen_strain() method's output.
        for eos in self.eos:
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
        return {eos: {} for eos in self.eos}
    
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
        for eos, key in self.unroll_strain_indices():
            times, hp, hc = self.strains[eos][key]
            projected = project_et(
                hp, hc, parameters=project_pars, sf=self.sample_rate, 
                nfft=2*self.sample_rate, detector=detector
            )
            self.strains[eos][key] = np.stack([times, projected])
        self.strains_history['projected'] = deepcopy(self.strains)
    
    def convert_to_IS_units(self) -> None:
        """Convert strains from geometrized to IS units."""
        
        if self.units == 'IS':
            raise RuntimeError("data already in IS units")

        for eos, key in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains[eos][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains[eos][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'IS'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.
    
    def convert_to_scaled_geometrized_units(self) -> None:
        """Convert strains from IS to geometrized units."""
        
        if self.units == 'geometrized':
            raise RuntimeError("data already in geometrized units")

        for eos, key in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains[eos][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains[eos][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)

        self.units = 'geometrized'
        # NOTE: No need to save a copy into 'strains_history' since the
        # operation is reversible.

    def export_strains_to_hdf5(self, file: str):
        """Save strains and their metadata to a HDF5 file.
        
        CAUTION! THIS DOES NOT SAVE THE ENTIRE INSTANCE!
        
        """
        with h5py.File(file, 'w') as hf:
            save_to_hdf5_recursive(
                data_dict=self.strains, metadata_dict=self.metadata, h5_group=hf
            )


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
        Chosen PSD of the ET detector. If NDArray, it must have shape
        (2, psd_length) so that:
            psd[0] = time samples
            psd[1] = psd samples

    """
    def __init__(self,
                 clean_dataset: CleanDataset, *,
                 psd: np.ndarray | Callable,
                 noise_length: int,
                 freq_cutoff: int | float,
                 freq_butter_order: int | float,
                 rng: np.random.Generator | int):
        """Initialize the injected dataset from a clean dataset.

        Relevant attributes are inherited from the CleanDataset instance.
        
        WARNING: This does not perform the injections! For that call the method
        'gen_injections'.
        
        """
        # Inherit clean strain instance attributes.
        self.eos = clean_dataset.eos
        self.sample_rate = clean_dataset.sample_rate
        self.strains_clean = deepcopy(clean_dataset.strains)
        self.metadata = deepcopy(clean_dataset.metadata)
        self.max_length = clean_dataset.max_length
        self.units = clean_dataset.units
        self.distance = clean_dataset.distance

        # Noise instance and related attributes.
        self.psd, self._psd = self._setup_psd(psd)
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
        return {eos: {key: {} for key in self.metadata[eos]} for eos in self.eos}
    
    def gen_injections(self, snr: int | float | list, pad: int = 0) -> None:
        """Inject all strains in simulated ET noise with the given SNR values.

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
        for eos, key in unroll_nested_dictionary_keys(self.strains_clean):
            gw_clean = self.strains_clean[eos][key]
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
                self.strains[eos][key][snr_] = np.stack([new_times, injected])
        
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
        for eos, key, snr in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains[eos][key][snr][0] *= mass * MSUN_SEC
            # Strain
            self.strains[eos][key][snr][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_clean_to_IS_units(self) -> None:
        for eos, key in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains_clean[eos][key][0] *= mass * MSUN_SEC
            # Strain
            self.strains_clean[eos][key][1:] *=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_to_scaled_geometrized_units(self) -> None:
        for eos, key, snr in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains[eos][key][snr][0] /= mass * MSUN_SEC
            # Strain
            self.strains[eos][key][snr][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)
    
    def _convert_strain_clean_to_scaled_geometrized_units(self) -> None:
        for eos, key in self.unroll_strain_indices():
            mass = self.metadata[eos][key]['mass']
            # Time
            self.strains_clean[eos][key][0] /= mass * MSUN_SEC
            # Strain
            self.strains_clean[eos][key][1:] /=  mass * MSUN_MET / (self.distance * MPC_MET)


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
