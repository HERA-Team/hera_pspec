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

        We can only set values on the slice level. No setting of values of slices
        is allowed!
        """
        # shape is a tuple
        if not isinstance(shape, tuple):
            raise ValueError("shape must be a tuple.")
        for m in shape:
            if not isinstance(shape, (int, ))
        self.shape = shape
        self.fill_value = fill_value
        self.ndims = len(self.shape)
        self.array = {}

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
        if not len(akey) > 0:
            for a in akey:
                if not (isinstance(a, slice) and a[0] is None and a[1] is None and a[2] is None):
                    raise ValueError("CompressedArray does not support setting values within slices!")
        if key in self.array:
            if not np.all(np.isclose(value, self.array(key))):
                for k in self.array.keys():
                    if np.all(np.isclose(value, self.array(key))):

                        break

        # akey must be empty!

        # if all array dims are unspecified or filled
        # then iterate through keys until we get an equal array.
        # if no equal/close array is found, initialize a new array.

        # if only some array dims are unspecified, find appropriate
        # slice, if it exists and fill these dims
        # if this slice does not exist, intialize a slice with fill_value.
        # and set values.

        # if psuedo dims are



    def mean(axis=None):
        """
        Take average of data
        """

    def to_numpy_array():

    def __eq__():
