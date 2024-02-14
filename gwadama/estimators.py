import numpy as np



def find_merger(h: np.ndarray) -> int:
    """Estimate the index position of the merger in the given strain.
    
    This could be done with a better estimation model, like a gaussian in
    the case of binary mergers. However for our current project this does not
    make much difference.
    
    """
    return np.argmax(np.abs(h))