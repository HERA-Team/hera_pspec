import numpy as np

# Class for storing high dimensional arrays with many copies of the same slice.
class CompressedArray():
    def __init__(shape=None, psuedo_dims=None, fill_value=0.0, rtol=1e-5, atol=1e-8, dtype=np.float64):
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
        self.shape = shape
        self.fill_value = fill_value
        if self.shape is not None:
            self.ndims = len(self.shape)
        else:
            self.ndims = None
        self.array = {}
        self.pdims = psuedo_dims
        self.npdims = len(self.pdims)
        # initialize array.
        self.nadims = self.ndims - self.npdims
        adims = []
        ashape = []
        if self.ndims is not None and self.psuedo_dims is not None:
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
                psuedo_key.append(int(k)) # always use ints for psuedo dimensions.
            else:
                array_key.append(k)
        return tuple(psuedo_key), tuple(array_key)




    def __setitem__(self, key, value):
        """set value or slice in compressed array

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
                # if we emptied this key group, then remove it and decrement all groups greater then it
                # so that indices are always spaced by 1 and range from 0 to nunique where nunique-1 where nunique is
                # the number of unique slices.
                if len(self.key_groups[this_kg]) == 0:
                    for kind in range(this_kg+1, self.nunique):
                        self.key_groups[kind-1] = self.key_groups.pop(kind)
                    self.nunique -= 1
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
        if not np.all(pkey>=0):
            raise NotImplementedError("Do not yet support negative indices for psuedo dimensions.")
        if pkey in self.array:
            if len(akey) > 0:
                return self.array[pkey][akey]
            else:
                return self.array[pkey]
        else:
            return np.ones(self.ashape, dtype=self.dtype) * self.fill_value

    def __eq__(self, other):
        """
        overloaded equality
        """
        equal = True
        # check that has same number of unique elements.
        equal = equal and list(self.key_groups.keys()) == list(other.key_groups.keys())
        # check that all slices are close
        for kg in self.key_groups:
            equal = equal and np.isclose(self.array[self.key_groups[kg][0]], self.array[self.key_groups[kg][0]],
                                         atol=self.atol, rtol=self.rtol)
        # check all attrs.
        equal = equal and self.fill_value == other.fill_value
        equal = equal and self.ndims == other.ndims
        equal = equal and self.pdims == other.pdims
        equal = equal and self.npdims == other.npdims
        equal = equal and self.nadims == other.nadims
        equal = equal and self.adims == other.adims
        equal = equal and self.ashape == other.ashape
        equal = equal and self.atol == other.atol
        equal = equal and self.rtol == other.rtol
        equal = equal and self.dtype == other.dtype
        equal = equal and self.nunique == other.nunique
        # check that key groups are the same when sorted by for column and then second column.
        for kg in self.key_groups:
            this_kg = np.asarray(self.key_groups[kg], dtype={'names':(str(a) for a in range(self.npdims)),
                                                             'formats':(str(np.integer) for a in range(self.npdims))})
            other_kg = np.asarray(other.key_groups[kg],dtype={'names':(str(a) for a in range(self.npdims)),
                                                             'formats':(str(np.integer) for a in range(self.npdims))})
            this_kg.sort(this_kg, order=[str(a) for a in range(self.npdims)])
            other_kg.sort(this_kg, order=[str(a) for a in range(self.nadims)])
            equal = equal and this_kg == other_kg
        return equal

    def write_to_group(self, group, label_stem):
        """
        write compressed array to hdf5 group

        Parameters
        ----------
        group: HDF5 group
            The handle of the HDF5 group that the compressed array should be written to

        label_stem: string stem for label of data group.

        """
        for kg in self.key_groups:
            if len(self.key_groups[kg]) > 0:
                k0 = self.key_groups[kg][0]
                group.create_dataset(label_stem + ".data_kg{}".format(kg),
                                     data=self.array[k0],
                                     dtype=self.dtype)
                group.create_dataset(label_stem + ".keys_kg{}".format(kg),
                                     data=np.asarray(self.key_groups[kg]),
                                     dtype=numpy.integer)
                group.attr[label_stem + ".fill_value"] = str(self.fill_value)
                group.attr[label_stem + ".ndims"] = str(self.ndims)
                group.attr[label_stem + ".pdims"] = str(self.pdims)
                group.attr[label_stem + ".npdims"] = str(self.npdims)
                group.attr[label_stem + ".nadims"] = str(self.nadims)
                group.attr[label_stem + ".adims"] = str(self.adims)
                group.attr[label_stem + ".ashape"] = str(self.ashape)
                group.attr[label_stem + ".atol"] = str(self.atol)
                group.attr[label_stem + ".rtol"] = str(self.rtol)
                group.attr[label_stem + '.dtype'] = str(self.dtype)
                group.attr[label_stem + '.nunique'] = str(self.nunique)

    def read_from_group(self, group, label_stem):
        """Read in compressed array from HDF5 group

        Parameters
        ----------
        group: HDF5 group
            The handle of the HDF5 group that the compressed array should be written to

        label_stem: string stem for label of data group.

        """
        self.dtype = np.dtype(group.attr[label_stem + '.dtype'])
        self.rtol = float(group.attr[label_stem + '.rtol'])
        self.atol = float(group.attr[label_stem + '.atol'])
        self.ashape = (int(a) for a in group.attr[label_stem + ".ashape"][1:-1].split(','))
        self.adims = (int(a) for a in group.attr[label_stem + ".adims"][1:-1].split(','))
        self.nadims = (int(a) for a in group.attr[label_stem + ".nadims"][1:-1].split(','))
        self.npdims = (int(a) for a in group.attr[label_stem + '.npdims'][1:-1].split(','))
        self.pdims = (int(a) for a in group.attr[label_stem + '.pdims'][1:-1].split(','))
        self.ndims = len(self.pdims) + len(self.adims)
        if self.dtype == np.float64 or self.dtype == np.float:
            self.fill_value = np.float(group.attr[label_stem + '.fill_value'])
        elif self.dtype == np.complex128 or self.dtype == np.complex:
            self.fill_value = np.complex(group.attr[label_stem + '.fill_value'])
        elif self.dtype = np.integer:
            self.fill_value = np.integer(group.attr[label_stem + '.fill_value'])
        elif self.dtype == np.bool
            self.fill_value = np.bool(group.attr[label_stem + '.fill_value'])
        self.nunique = int(group.attr[label_stem+'.nunique'])
        # now read in data.
        for kg in range(self.nunique):
            data = group[label_stem + ".data_kg{}".format(kg)]
            keys = tuple(group[label_stem + '.keys_kg{}'.format(kg)])
            for k in keys:
                self.__setitem__(k, data)
        # all good!
