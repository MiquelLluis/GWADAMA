"""dictools.py

Collection of utility functions related to nested Python dictionaries.

"""
# from copy import deepcopy  # Lazy import
from typing import Any, Iterable

import numpy as np


def unroll_nested_dictionary_keys(dict_: dict, max_depth: int|None = None) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    Useful to iterate over all keys of a nested dictionary without having to
    use multiple loops.

    Parameters
    ----------
    dictionary: dict
        Nested dictionary.
    
    max_depth: int, optional
        If specified, it is the number of layers to dig in to at most in
        the nested dictionary.
        If only the first layer is desired (no recursion at all), `max_depth=1`.
    
    Returns
    -------
    : list
        Unrolled combinations of all keys of the nested dictionary.
    
    """
    return __unroll_nested_dictionary_keys(dict_, max_depth=max_depth)


def __unroll_nested_dictionary_keys(dict_: dict,
                                    *,
                                    max_depth: int|None = None,
                                    current_keys: list|None = None,
                                    current_depth: int = 1) -> list:
    """Returns a list of all combinations of keys inside a nested dictionary.
    
    This is the recursive function. Use the main function.
    
    """
    if current_keys is None:
        current_keys = []

    unrolled_keys = []

    for key, value in dict_.items():
        new_keys = current_keys + [key]

        if isinstance(value, dict) and (max_depth is None or current_depth < max_depth):
            # Go down next depth layer.
            unrolled_keys += __unroll_nested_dictionary_keys(
                value, max_depth=max_depth, current_keys=new_keys, current_depth=current_depth+1
            )
        else:
            unrolled_keys.append(new_keys)

    return unrolled_keys


def get_value_from_nested_dict(dict_: dict, keys: Iterable[Any]) -> Any:
    """Access a value from a nested dictionary using a sequence of keys.

    Parameters
    ----------
    dict_ : dict
        A dictionary which may contain further nested dictionaries.
    keys : iterable
        A sequence of keys that defines the path to the target value.

    Returns
    -------
    Any
        The value located at the nested key path.

    Warnings
    --------
    The returned value is the original object stored in the dictionary and can
    be modified in-place. Use this behaviour with caution.

    Notes
    -----
    This function *only* traverses Python `dict` objects. If a path segment
    encounters a non-dict (e.g., a NumPy array), a `KeyError` is raised.
    This prevents accidental array indexing when a path is malformed.
    """
    if not isinstance(dict_, dict):
        raise TypeError("'dict_' must be a dictionary")

    cur: Any = dict_
    path = list(keys)

    # Empty path: return the dictionary itself (consistent with traversal)
    if not path:
        return cur

    for i, key in enumerate(path):
        if not isinstance(cur, dict):
            raise KeyError(
                f"Non-dict encountered at depth {i} while traversing {path!r}; "
                f"current object type is {type(cur).__name__}"
            )
        if key not in cur:
            avail = list(cur.keys())
            raise KeyError(
                f"Key {key!r} not found at depth {i}; available keys: {avail!r}"
            )
        cur = cur[key]

    return cur


def set_value_to_nested_dict(dict_: dict,
                             keys: Iterable[Any],
                             value: Any,
                             *,
                             add_missing_keys=False) -> None:
    """Set a value in a nested dictionary using a sequence of keys.

    Parameters
    ----------
    dict_: dict
        Target nested dictionary.

    keys: iterable
        Sequence of keys necessary to reach the element inside the nested
        dictionary.

    value: Any
        Value to set at the target location.

    add_missing_keys: bool
        If True, missing intermediate keys are created as empty dicts.
        If False (default), a missing key raises `KeyError`.
    
    Raises
    ------
    TypeError
        If `dict_` is not a dictionary.
    KeyError
        If an intermediate key is missing (and `add_missing_keys` is False),
        or if a non-dict object is encountered before the final key.

    Notes
    -----
    - This function *only* creates/traverses plain `dict` containers.
    - It refuses to descend into non-dict objects, avoiding accidental array
    indexing or attribute misuse mid-path.
    """
    if not isinstance(dict_, dict):
        raise TypeError("'dict_' must be a dictionary")

    path = list(keys)
    if not path:
        raise ValueError("'keys' must contain at least one element")
    
    cur: Any = dict_
    for i, key in enumerate(path[:-1]):
        if not isinstance(cur, dict):
            raise KeyError(
                f"Non-dict encountered at depth {i} while traversing {path!r}; "
                f"current object type is {type(cur).__name__}"
            )
        if key not in cur:
            if add_missing_keys:
                cur[key] = {}
            else:
                avail = list(cur.keys())
                raise KeyError(
                    f"Key {key!r} not found at depth {i}; available keys: {avail!r}"
                )
        cur = cur[key]

    # Final parent must be a dict
    if not isinstance(cur, dict):
        raise KeyError(
            f"Cannot set value at final parent (type {type(cur).__name__}); "
            f"expected a dictionary at depth {len(path)-1}"
        )
    cur[path[-1]] = value


def fill(dict_: dict, value, keys=None, deepcopy=False):
    """Fill an arbitrarily-depth nested dictionary with a value.

    Fill an arbitrarily-depth nested dictionary below the coordinates 'keys'
    with the value 'value'.

    The filling is performed inplace.

    Parameters
    ----------
    dict_: dict
        Nested dictionary.

    value: Any

    keys: iterable, optional
        Starting layers from where to fill the dictionary, if only a subset
        of the whole 'dict_' is desired to be filled.

    deepcopy: bool
        If True, each instance of 'value' will be a copy. By default all
        elements reference the same 'value'.

    """
    if keys is not None:
        # Get to the target layer.
        for key in keys:
            dict_ = dict_[key]

    __fill(dict_, value, deepcopy=deepcopy)


def __fill(dict_: dict, value, deepcopy: bool = False):
    """Fill an arbitrarily-depth nested dictionary with a value.

    This is the recursive function. Use the main function.

    """
    for key in dict_:
        if isinstance(dict_[key], dict):
            __fill(dict_[key], value, deepcopy=deepcopy)
        else:
            if deepcopy:
                try:
                    dict_[key] = value.copy()
                except AttributeError:
                    from copy import deepcopy as dc
                    dict_[key] = dc(value)
            else:
                dict_[key] = value



def replicate_structure(dict_: dict) -> dict:
    """Create a new nested dictionary with the same structure as the input.

    Values of the new dictionary are set to None.

    """
    if not isinstance(dict_, dict):
        return None

    replicated_dict = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            replicated_dict[key] = replicate_structure(value)
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


def dict_to_stacked_array(dict_: dict, target_length: int|None = None) -> tuple[np.ndarray, list]:
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


def find_parent_key_of_nested_key(dict_, key: int|str) -> int | str:
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


def __flatten_nested_dict(dict_, parent_keys=()):
    """Flatten recursively 'dict_in'.
    
    Here is where the actual flattening happens, using recursion.
    
    """
    flattened_dict = {}
    for k, v in dict_.items():
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


def get_first_value(dict_):
    """Get the first value in a nested dictionary.
    
    If `dict_` is not a python dictionary, the argument is returned as is.

    Returns
    -------
    value : Any
        Next value in `dict_`.
    
    """
    while isinstance(dict_, dict) and dict_:
        dict_ = next(iter(dict_.values()))
    return dict_


def get_number_of_elements(dict_):
    """Get the number of elements in a nested dictionary.

    Parameters
    ----------
    dict_ : dict
        Nested dictionary.

    Returns
    -------
    number : int
        Number of elements in the nested dictionary.

    """
    if not isinstance(dict_, dict):
        raise TypeError("'dict_' must be a dictionary")

    number = 0
    for value in dict_.values():
        if isinstance(value, dict):
            number += get_number_of_elements(value)
        else:
            number += 1
    
    return number


def get_types(d: dict):
    """Get all element types in a nested dictionary.

    Return all unique non-dict types from values in a (possibly nested)
    dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    types : set
        Set of non-dict Python types.

    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dictionary for 'd', got {type(d).__name__} instead.")

    types = set()
    stack = [d]

    while stack:
        current = stack.pop()
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)  # keep traversing
            else:
                types.add(type(value))  # record type, don't traverse further

    return types
