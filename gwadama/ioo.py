import h5py
import yaml



def load_config(file):
    with open(file) as f:
        cfg = yaml.safe_load(f)
    
    return cfg


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
            h5_group=hf, data=self.strains, metadata=self.metadata
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
