"""Input & Output Operations

Everything related to input data retrieval and output files.

"""
from pathlib import Path
from shutil import rmtree

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .units import *


# Note: 'watpy' is an optional dependency and is imported lazily inside CoReManager.
# This allows users to use CLAWDIA without requiring 'watpy' unless they need CoReManager.




class CoReManager:
    """Manage easily usual tasks of Watpy's CoRe_db instances.

    Ad-hoc manager to automate and make easier usual tasks in Watpy, including
    metadata and strain downloads, and spatial projection.

    ATTRIBUTES
    ----------
    cdb : watpy.coredb.coredb.CoRe_db
        Instance of CoRe_db from which everything is managed.

    db_path : Path
        Folder where is or will be stored CoRe database.
        Refer to 'watpy.coredb.coredb.CoRe_db' for more details.

    eos : set
        All EOS found available.

    metadata : DataFrame
        Metadata from all simulations in 'cdb' collected in a single DF.

    downloaded : dict
        3-Level dictionary containing the path to existing strains (saved as
        TXT files), their eccentricity and radius of extraction.
        Tree-format:
            txt_files[simkey][run] = {
                'file': 'path/to/file.txt',
                'eccentricity': ecc,
                'r_extraction': rext
            }
        NOTE: ONLY KEEPS 1 RUN PER SIMULATION FOR NOW

    """
    fields_float = [
        'id_mass',
        'id_rest_mass',
        'id_mass_ratio',
        'id_ADM_mass',
        'id_ADM_angularmomentum',
        'id_gw_frequency_Hz',
        'id_gw_frequency_Momega22',
        'id_kappa2T',
        'id_Lambda',
        'id_eccentricity',
        'id_mass_starA',
        'id_rest_mass_starA',
        'id_mass_starB',
        'id_rest_mass_starB'
    ]
    header_gw_txt = "u/M:0 Reh/M:1 Imh/M:2 Redh/M:3 Imdh/M:4 Momega:5 A/M:6 phi:7 t:8"

    def __init__(self, db_path):
        """Initialize the manager.

        PARAMETERS
        ----------
        db_path : str
            Folder where is or will be stored CoRe database.
            Refer to 'watpy.coredb.coredb.CoRe_db' for more details.

        """
        try:
            import watpy  # Lazy import
        except ImportError:
            raise ImportError(
                "The 'watpy' package is required for CoReManager but is not installed."
            )

        self.db_path = Path(db_path)
        self.cdb = watpy.coredb.coredb.CoRe_db(db_path)
        self.metadata = self._gen_metadata_dataframe()
        self.eos = set(self.metadata['id_eos'])
        self.downloaded = self._look_for_existing_strains()

    def __repr__(self):
        return str(self.metadata)

    def __getitem__(self, i):
        return self.metadata[i]

    def __len__(self):
        return len(self.metadata)

    def show(self, key, to_float=False, to_file=None):
        return self.cdb.idb.show(key, to_float=False, to_file=None)

    def filter_by(self, key, value):
        return self.metadata[self.metadata[key] == value]

    def filter_multiple(self, filters):
        md = self.metadata.copy()
        for k, v in filters:
            md = md[md[k] == v]

        return md

    def count_runs(self, filters=[]):
        """Count total number of runs in the database.

        Parameters
        ----------
        filters : list of lists
            Format: [[key0, value0], [key1, value1], ...]

        """
        md = self.filter_multiple(filters)
        counts = 0
        for ind in md.index:
            runs = md.loc[ind].available_runs
            if runs is not None:
                counts += len(runs.split(', '))

        return counts

    def download_mode22(self, simkeys, keep_h5=False, overwrite=False,
                         prot='https', lfs=False, verbose=True):
        """Download ONLY the optimum strains Rh_22.

        Downloads each simulation, keeps the strains with the lowest
        eccentricity and highest extraction point 'r' in a TXT file, updates
        the database 'self.downloaded', and (optional) removes the original
        HDF5 file from CoRe to free up space.

        Parameters
        ----------
        simkeys : list
            List of simulation keys ('db_keys' in watpy) to download.

        keep_h5 : bool
            If False (default) removes the HDF5 file downloaded by watpy.

        overwrite : bool
            If False (default) and a certain simulation in 'simkeys' is already
            present in 'self.downloaded', skip it. Otherwise downloads
            everything again.

        verbose : bool
            If True (default), print which simulations are downloaded and which
            are skipped.

        prot, lfs :
            Refer to 'watpy.coredb.coredb.CoRe_db.sync'.

        """
        for skey in tqdm(simkeys):
            if (not overwrite) and (skey in self.downloaded):
                if verbose: print(f"{skey} already downloaded, skipping.")
                continue

            self.cdb.sync(dbkeys=[skey], prot=prot, lfs=lfs, verbose=False)

            # Get the gw data with lowest eccentricity and at the highest 'r'
            # of extraction.
            runkey, ecc = self.get_runkey_lowest_eccentricity(skey)
            run = self.cdb.sim[skey].run[runkey]
            data = run.data.read_dset()['rh_22']
            rext_key, rext = self._get_highest_r_extraction_(data)
            gw_data = data[rext_key]

            # Save gw data as TXT.
            ofile = Path(run.path) / f'Rh_l2_m2_r{rext:05d}.txt'
            np.savetxt(ofile, gw_data, header=self.header_gw_txt)

            # Update download database.
            self.downloaded[skey] = {}  # ONLY 1 RUN PER SIM FOR NOW
            self.downloaded[skey][runkey] = {
                'file': ofile,
                'eccentricity': ecc,
                'r_extraction': rext
            }

            if not keep_h5: self._clean_h5_data(skey)
            if verbose: print(f"{skey} downloaded.")

    def load_sim(self, skey):
        """Load a previously downloaded gw simulation."""
        
        meta_sim = self.downloaded[skey]
        file = next(iter(meta_sim.values()))['file']
        gw_data = np.loadtxt(file)

        return gw_data

    @staticmethod
    def sw_Y22(i, phi):
        """Spin-weighted spherical harmonic mode lm = 22.

        Ref: Ajith et al., 2011

        Parameters:
        -----------
        i : float
            Inclination angle from the z-axis.

        phi : float
            Phase angle.

        """
        return np.sqrt(5/(64*np.pi)) * (1 + np.cos(i))**2 * np.exp(2j*phi)

    def gen_strain(self, skey, distance, inclination, phi):
        """Build strain from time-domain mode 22 in mass rescaled, geom. units.
        
        Parameters:
        -----------
        skey : str
            Key (database_key) of the simulation.

        distance : float
            Distance to the source in Mpc.
        
        inclination, phi: float
            Angle positions.

        Returns:
        --------
        time : ndarray(float)
            Time points in seconds.
        
        hplus, hcross : ndarray(float)
            Polarizations of the rescaled strain.
        
        """
        gw_data = self.load_sim(skey)
        mass = self.metadata.id_mass.loc[skey]
        u_M = gw_data[:,0]
        Rh = gw_data[:,1] - 1j*gw_data[:,2]

        # Convert time.
        time = u_M * mass * MSUN_SEC

        # Genearte Strain polarizations.
        sY22 = self.sw_Y22(phi, inclination)
        amplitude_prefactor = mass * MSUN_MET / (distance * MPC_MET)
        h = amplitude_prefactor * Rh * sY22
        hplus = h.real
        hcross = -h.imag
        
        return time, hplus, hcross

    def get_runkey_lowest_eccentricity(self, skey):
        """Find the run with the lowest eccentricity for a given simulation.

        Return the key of the run and the value of its eccentricity for which
        this parameter is the lowest among all runs of the 'skey' simulation.

        If a simulation has multiple runs with the same eccentricity
        (typically all values set to 0 or NAN) it will pick the first run in
        the list.

        If there are one or more runs with eccentricity = NAN, the first one
        will be returned.

        Parameters
        ----------
        skey : str
            Key (database_key) of the simulation.

        Returns
        -------
        run_key : str
            Key of the run.
        ecc : float
            Eccentricity of the run.

        """
        runs = self.cdb.sim[skey].run.copy()
        run_key = 'R01'
        ecc = self.cast_to_float(runs.pop('R01').md.data['id_eccentricity'])

        if np.isnan(ecc):
            return run_key, ecc

        # Look for the minimum eccentricity, or the first one to be NAN.
        for rkey, run in runs.items():
            ecc_i = self.cast_to_float(run.md.data['id_eccentricity'])
            if ecc_i < ecc:
                run_key = rkey
                ecc = ecc_i
            elif np.isnan(ecc_i):
                run_key = rkey
                ecc = ecc_i
                break

        return run_key, ecc

    @staticmethod
    def cast_to_float(string):
        """Cast a string to float, considering '' also a NaN."""

        if string in ['', None]:
            n = np.nan
        else:
            n = float(string)

        return n

    def _clean_h5_data(self, skey):
        """Remove HDF5 files from a downloaded 'skey' simulation.

        Parameters
        ----------
        skey : str
            Simulation key.

        """
        root = Path(self.cdb.sim[skey].path)
        # Remove HDF5 files.
        files = root.glob('*/*.h5')
        for file in files:
            file.unlink()
        # Remove .git folder.
        folder = root / '.git'
        rmtree(folder)

    def _gen_metadata_dataframe(self):
        idb = self.cdb.idb
        key_list = idb.dbkeys
        metalist = [core_md.data for core_md in idb.index]
        md = pd.DataFrame(metalist, index=key_list)
        # Convert data types of the selected columns:
        for field in self.fields_float:
            mask = (md[field] == 'NAN') | (md[field] == '')
            md[field].values[mask] = np.nan
            md[field] = md[field].astype(float)

        return md

    def _get_highest_r_extraction_(self, extractions):
        """Return the key and value of 'r' of the gw with the highest 'r'.

        It also includes the case when the value of 'r' in the data is Inf
        instead of a number.

        Parameters
        ----------
        extractions : dict
            Channel 'rh_22' returned by CoRe_h5.read_dset().

        """
        rext_key = max(extractions.keys())
        if 'Inf' in rext_key:
            rext = 99999  # Highest number with 5 figures, to represent infinity.
        else:
            rext = int(rext_key[-9:-4])

        return rext_key, rext

    def _look_for_existing_strains(self):
        """Strains that were already extracted from CoRe's HDF5 files.

        Save their paths, alongside the eccentricity and radius of extraction,
        in a dictionary tree by simulation key and run.

        Returns
        -------
        txt_files : dict
            3-Level dictionary containing the path to existing strains
            (extracted as TXT files), their eccentricity and radius of
            extraction. Tree-format:
                txt_files[simkey][run] = {
                    'file': 'path/to/file.txt',
                    'eccentricity': ecc,
                    'r_extraction': rext
                }

        """
        txt_files = {}
        for file in self.db_path.rglob('Rh*.txt'):
            key = file.parts[-3].replace('_', ':')
            run = file.parts[-2]
            ecc = self._read_eccentricity(file.parent/'metadata.txt')
            rext = int(file.stem[-5:])
            
            if key not in txt_files:
                txt_files[key] = {}
            
            # If there are multiple files of the same sim and run, only keep
            # the highest extraction point available.
            elif run in txt_files[key]:
                r0 = int(txt_files[key][run]['file'].stem[-5:])
                if rext < r0:
                    continue

            txt_files[key][run] = {
                'file': file,
                'eccentricity': ecc,
                'r_extraction': rext
            }

        return txt_files

    def _read_eccentricity(self, file):
        """Get the value of eccentricity from a metadata file."""

        with open(file) as f:
            for line in f:
                if 'id_eccentricity' in line:
                    break
            else:
                raise EOFError("no value of eccentricity found in the file")
        ecc_txt = line.split()[-1]
        try:
            ecc = float(ecc_txt)
        except ValueError:
            ecc = np.nan

        return ecc


def save_to_hdf5(file: str, *, data: dict, metadata: dict) -> None:
    """Save nested dictionary of NumPy arrays to HDF5 file with metadata.

    PARAMETERS
    ----------
    file: str
        Path to the HDF5 file.

    data: dict
        Nested dictionary containing NumPy arrays.
    
    metadata: dict
        Nested dictionary containing metadata for arrays.

   
    EXAMPLE
    -------
    ```
    data = {
        'group1': {
            'array1': np.array([1, 2, 3]),
            'array2': np.array([4, 5, 6])
        },
        'group2': {
            'array3': np.array([7, 8, 9]),
            'array4': np.array([10, 11, 12])
        }
    }
    metadata = {
        'group1': {
            'array1': {'description': 'Example data'},
            'array2': {'description': 'Another example'}
        },
        'group2': {
            'array3': {'unit': 'm/s^2'},
            'array4': {'unit': 'kg'}
        }
    }
    save_to_hdf5('output.h5', data=data, metadata=metadata)
    ```
    
    """
    with h5py.File(file, 'w') as hf:
        _save_to_hdf5_recursive(
            h5_group=hf, data=data, metadata=metadata
        )
    

def _save_to_hdf5_recursive(*,
                            h5_group: h5py.Group,
                            data: dict,
                            metadata: dict) -> None:
    """Save nested dictionary of NumPy arrays to HDF5 group with metadata.

    Save nested dictionary of NumPy arrays to HDF5 group with metadata
    formatted in the same way as the data.

    
    PARAMETERS
    ----------
    h5_group: h5py.Group
        H5PY Group into which store the data and optional metadata.

    data_dict: dict
        Nested dictionary containing NumPy arrays.
    
    metadata_dict: dict
        Nested dictionary containing metadata for arrays.

    """
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = h5_group.create_group(key)
            _save_to_hdf5_recursive(
                data=value, metadata=metadata[key], h5_group=subgroup
        )
        else:
            dataset = h5_group.create_dataset(key, data=value)
            for metadata_key, metadata_value in metadata[key].items():
                dataset.attrs[metadata_key] = metadata_value


def save_experiment_results(file: str,
                            *,
                            parameters: dict,
                            results: dict,
                            sep=';',
                            start_index: int = 0,
                            return_dataframe=False):
    """Save (update) experiment results to a CSV using pandas.DataFrame.

    Parameters and results are saved as columns in a Pandas DataFrame, in a
    new row of an existing CSV file, or as the first row if the file does not
    exists. Each row represents an experiment.
    Parameters' keys may change depending on the experiment; when adding a new
    key or omitting one, the corresponding place in the CSV file will be filled
    with NaN.

    Results are inserted at the beginning of the DataFrame's columns, and their
    keys must not change between experiments.

    If there is a complex parameter structure, it is recommended to provide
    a flattened version with new and simpler keys.

    Parameters
    ----------
    file : str
        Path to the CSV file.
        If the file already exists, loads it and appends the experiment data
        to the end of the DataFrame, and updates the file.
        If the file does not exist, creates it.
    
    parameters : dict
        Experiment settings and hyper-parameters.
        They must be registered in a flat dictionary.
    
    results : dict
        Experiment results, tipically the LOSS overall statistics.
        For example:
        ```
        results = {
            'loss max': 0.3,
            'loss_mean': 0.1,
            'loss_median': 0.08,
            'loss_min': 0.001
        }
        ```
        They will be inserted at the beginning of the DataFrame's columns.
    
    sep : str
        Column separator in the CSV file. ';' by default.
    
    start_index : int
        Index of the first row in the experiment (and the CSV file) when
        creating a new file. If the file already exists, this is ignored.
        0 by default.
    
    return_dataframe : bool
        If True, return the DataFrame instance. False by default.
    
    Returns
    -------
    df : pandas.DataFrame, optional
        The pandas DataFrame instance, only returned if 'return_dataframe'
        is True.
    
    """
    data = {**results, **parameters}

    try:
        df = pd.read_csv(file, sep=sep, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame([], columns=data.keys())
        df.index.name = 'Experiment'
        i_exp = start_index
    else:
        # Make sure input results keys are the same as the ones in the file.
        columns_old = list(df.columns)
        results_keys = list(results.keys())
        i_sep = columns_old.index(results_keys[-1]) + 1
        if set(results_keys) != set(columns_old[:i_sep]):
            raise ValueError('Experiment results keys must not change between experiments.')

        i_exp = df.index[-1] + 1


    for k, v in data.items():
        df.at[i_exp, k] = v
    
    df.to_csv(file, sep=sep)

    if return_dataframe:
        return df
