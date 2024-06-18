"""dictools.py

Collection of utility functions related to nested Python dictionaries.

"""
import numpy as np


def unroll_nested_dictionary_keys(dictionary: dict, max_depth: int = None) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    Useful to iterate over all keys of a nested dictionary without having to
    use multiple loops.

    Parameters
    ----------
    dictionary: dict
        Nested dictionary.
    
    max_depth: int, optional
        If specified, it is the number of layers to dig in to at most in
        the nested 'strains' dictionary.
        If only the first layer is desired (no recursion at all), `max_depth=1`.
    
    Returns
    -------
    : list
        Unrolled combinations of all keys of the nested dictionary.
    
    """
    return __unroll_nested_dictionary_keys(dictionary, max_depth=max_depth)


def __unroll_nested_dictionary_keys(dictionary: dict,
                                    *,
                                    max_depth: int,
                                    current_keys: list = None,
                                    current_depth: int = 1) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    This is the recursive function. Use the main function.
    
    """
    if current_keys is None:
        current_keys = []

    unrolled_keys = []

    for key, value in dictionary.items():
        new_keys = current_keys + [key]

        if isinstance(value, dict) and (max_depth is None or current_depth < max_depth):
            # Go down next depth layer.
            unrolled_keys += __unroll_nested_dictionary_keys(
                value, max_depth=max_depth, current_keys=new_keys, current_depth=current_depth+1
            )
        else:
            unrolled_keys.append(new_keys)

    return unrolled_keys


def _get_value_from_nested_dict(dict_, keys: list):
    value = dict_
    for key in keys:
        if not isinstance((value:=value[key]), dict) and not hasattr(value, '__iter__'):
            raise ValueError("the nested dictionary shape does not match with the input key sequence")

    return value


def set_value_to_nested_dict(dict_, keys, value, add_missing_keys=False):
        """Set a value to an arbitrarily-depth nested dictionary.

        Parameters
        ----------
        dict_: dict
            Nested dictionary.

        keys: iterable
            Sequence of keys necessary to get to the element inside the nested
            dictionary.

        value: Any

        add_missing_keys: bool
            If True, missing keys (layers) will be added to the nested
            dictionary.
            
            CAUTION: if `add_missing_keys=True`, no KeyError will be raised.

        """
        for key in keys[:-1]:
            if key not in dict_:
                if add_missing_keys:
                    dict_[key] = {}
                else:
                    raise ValueError(
                        "the nested dictionary shape does not match with the input key sequence"
                    )
            
            dict_ = dict_[key]
        dict_[keys[-1]] = value


def _replicate_structure_nested_dict(input_dict: dict) -> dict:
    """Create a new nested dictionary with the same structure as the input.

    Values of the new dictionary are set to None.

    """
    if not isinstance(input_dict, dict):
        return None

    replicated_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            replicated_dict[key] = _replicate_structure_nested_dict(value)
        else:
            replicated_dict[key] = None

    return replicated_dict


def get_depth(dict_: dict) -> int:
    """Return the depth of the input nested dictionary.
    
    A simple (non-nested) dictionary has a depth of 1.
    Assumes a homogeneous nested dictionary, and only looks for the first
    element at each layer.
    
    """
    depth = 0
    while isinstance(dict_, dict):
        key = next(iter(dict_.keys()))
        dict_ = dict_[key]
        depth += 1
    
    return depth


def dict_to_stacked_array(dict_: dict, target_length: int = None) -> tuple[np.ndarray, list]:
    """Stack the arrays inside a dict() to a 2d-array.
    
    Given a NON-nested dict whose values are flat numpy arrays, with potentially different
    lengths, stacks them in a homogeneous 2d-array aligned to the left,
    zero-padding the remaining space.

    Parameters
    ----------
    dict_ : dict[str: np.ndarray]
        NON-nested Python dictionary containing numpy 1d-arrays.
    
    target_length : int, optional
        If given, defines the size of the second axis of the returned 2d-array.
        If omitted, the size will be equal to the longest array inside 'dict_'.
        Must be larger or equal than the longest array inside 'dict_'.
    
    Returns
    -------
    stacked_arrays : 2d-array
        Stacked arrays, with right zero-padding those original strains whose
        length were shorter.
    
    lengths : list
        Original length of each input array, following the same order as the
        first axis of 'stacked_arrays'.
    
    """
    if target_length is None:
        target_length = 0
        for array in dict_.values():
            l = len(array)
            if l > target_length:
                target_length = l
        if target_length == 0:
            raise ValueError(
                "no arrays with nonzero length were found inside 'dict_'"
            )

    stacked_arrays = np.zeros((len(dict_), target_length), dtype=float)
    lengths = []

    for i, array in enumerate(dict_.values()):
        l = len(array)
        pad = target_length - l
        if pad < 0:
            raise ValueError(
                "given 'target_length' is smaller than the longest array inside 'dict_'"
            )
        stacked_arrays[i] = np.pad(array, (0, pad))
        lengths.append(l)
    
    return stacked_arrays, lengths


def _find_level0_of_level1(dict_, key: int|str) -> int | str:
    """Finds the top level key containing the second level 'key'.

    Finds the key of the uppermost level of the nested 'dict_' which contains
    the second level entry with key 'key'.

    Parameters
    ----------
    dict_ : dict
        Nested dictionary.

    key : int | str
        Key of the second level of 'dict_'.
    
    Returns
    -------
    key_top : int | str
        Key of the uppermost level of 'dict_'.
    
    Raises
    ------
    ValueError
        If 'key' is not found in the second level of 'dict_'.
    
    """
    for key_top, level1 in dict_.items():
        if key in level1:
            return key_top
    else:
        raise ValueError(f"key '{key}' was not found inside the second level of the dictionary")


def flatten_nested_dict(dict_: dict) -> dict:
    """Turn any nested dictionary into a shallow (single level) one.
    
    Flatten a nested dictionary into a single level dictionary, keeping their
    keys as tuples.
    
    """
    return __flatten_nested_dict(dict_)


def __flatten_nested_dict(dict_in, parent_keys=()):
    """Flatten recursively 'dict_in'.
    
    Here is where the actual flattening happens, using recursion.
    
    """
    flattened_dict = {}
    for k, v in dict_in.items():
        key = parent_keys + (k,)
        if isinstance(v, dict):
            flattened_dict.update(__flatten_nested_dict(v, parent_keys=key))
        else:
            flattened_dict[key] = v
    
    return flattened_dict
    

def filter_nested_dict(dict_, condition, layer) -> dict:
    """Filter a layer of a nested dictionary.

    Filter a nested dictionary based on a condition applied to the keys of the
    specified layer.

    NOTE: Layer numbering begins with 0, as array-likes do; as God commands.

    Parameters
    ----------
    dict_ : dict
        The nested dictionary to be filtered.
    
    condition : callable
        The condition function to apply.
        Should take a single argument, the key, and return a boolean indicating
        wether to include its related value.
    
    layer : int
        The layer at which to apply the condition.
        1 corresponds to the top level, 2 to the second level, and so on.
        Default is 1.

    Returns
    -------
    : dict
        Filtered version of the nested dictionary.

    Caveats
    -------
    - The filtering does not alter the order of kept elements in 'dict_'.

    """
    def filter_layer(dictionary, current_layer):
        if current_layer == layer:
            return {k: v for k, v in dictionary.items() if condition(k)}

        return {k: filter_layer(v, current_layer + 1) if isinstance(v, dict) else v
                for k, v in dictionary.items()}

    return filter_layer(dict_, 0)


def get_next_item(dict_):
    """Get the next item in a nested dictionary.

    Returns
    -------
    value : Any
        Value of the next item in the dictionary.

    """
    if not isinstance(dict_, dict):
        raise TypeError("'dict_' must be a dictionary")

    try:
        value = next(iter(dict_.values()))
    except StopIteration:
        # Empty dictionary.
        return None

    if isinstance(value, dict):
        return get_next_item(value)
    return value
