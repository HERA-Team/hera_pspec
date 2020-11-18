import numpy as np

# Class for storing high dimensional arrays with many copies of the same slice.
class CompressedArray():
    def __init__(shape, psuedo_dims, init_value=0.0, rtol=1e-5, atol=1e-8, dtype=np.float64):
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
        self.pdims = psuedo_dims
        self.npdims = len(self.pdims)
        # initialize array.
        self.nadims = self.ndims - self.npdims
        adims = []
        ashape = []
        for m in range(self.ndims):
            if m not in psuedo_dims:
                adims.append(m)
                ashape.append(self.shape[m])
        self.adims = tuple(adims)
        self.ashape = tuple(ashape)
        self.atol = atol
        self.rtol = rtol
        self.dtype = dtype
        self.key_groups = {} # dictionary with lists of keys for each
                                 # integer indexed unique item.
        self.nunique = len(self.key_groups)


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
                    raise ValueError("CompressedArray does not allow setting values within existing slices!")
        if pkey in self.array:
            # determine group that pkey is in
            for kg in self.key_groups:
                if pkey in self.key_groups[kg]:
                    this_kg = kg
                    this_index = self.key_groups[kg].index(pkey)

            if not np.all(np.isclose(value, self.array[pkey], atol=self.atol, rtol=self.rtol)):
                exist = False
                for kg in self.key_groups:
                    k = self.key_groups[kg][0]
                    if np.all(np.isclose(value, self.array[k], atol=self.atol, rtol=self.rtol)):
                        self.array[pkey] = self.array[k]
                        self.key_groups[kg].append(self.key_groups[this_kg].pop(kind))
                        exists = True
                # if value does not exist, set slice to that value.
                if not exists:
                    self.array[pkey] = value.astype(self.dtype)
                    self.key_groups[self.nunique] = [self.key_groups[this_kg].pop(kind)]
                    self.nunique += 1
                # prune empty key groups
                if len(self.key_groups[this_kg]) == 0:
                    del self.key_groups[this_kg]
        else:
            # only support integer keys with same length of npdim for now
            if not len(pkey) == self.npdims:
                raise NotImplementedError("CompressedArray does not yet support partial keys!")
            for p in pkey:
                if not isinstance(p, (int, np.integer)):
                    raise NotImplementedError("CompressedArray does not yet support slicing of psuedo_dims!")
            self.array[pkey] = value
            self.key_groups[self.nunique] = [pkey]
            self.nunique += 1


    def __getitem__(self, key, value):
        """

        """
        pkey, akey = parse_key(key)
        # only support integer keys with same length of npdim for now
        if not len(pkey) == self.npdims:
            raise NotImplementedError("CompressedArray does not yet support partial keys!")
        for p in pkey:
            if not isinstance(p, (int, np.integer)):
                raise NotImplementedError("CompressedArray does not yet support slicing of psuedo_dims!")
        if pkey in self.array:
            if len(akey) > 0:
                return self.array[pkey][akey]
            else:
                return self.array[pkey]
