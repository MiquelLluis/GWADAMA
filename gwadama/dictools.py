"""dictools.py

Collection of utility functions related to nested Python dictionaries.

"""


def _unroll_nested_dictionary_keys(dictionary: dict, max_depth: int = None) -> list:
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