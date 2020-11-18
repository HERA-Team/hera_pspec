import numpy as np

# Class for storing high dimensional arrays with many copies of the same slice.
class CompressedArray():
    def __init__(shape, psuedo_dims, fill_value=0.0):
        """
        shape: tuple
            tuple of integers giving the shape of our the compressed array.
        psuedo_dims: list
            list of integers representing dimensions to be treated as "psuedo"
            in that they are actually dictionaries with redundant pointers to
            slices.
        """
        # shape is a tuple
        if not isinstance(shape, tuple):
            raise ValueError("shape must be a tuple.")
        for m in shape:
            if not isinstance(shape, (int, ))
        self.shape = shape
        self.fill_value = fill_value
        self.ndims = len(self.shape)

    def _parse_key(key):
        """
        Function that parses array key into an array component
        and a psuedo component.
        """
        # go through dimensions
        psuedo_key = []
        array_key = []
        for knum,k in enumerate(key):
            if knum in psuedo_dims:
                psuedo_key.append(k)
            else:
                array_key.append(k)
        return tuple(psuedo_key), tuple(array_key)




    def __setitem__(self, key, value):
        """

        """
        pkey, akey = parse_key(key)
        # if psuedo dims



    def mean(axis=None):
        """
        Take average of data
        """

    def to_numpy_array():

    def __eq__():
