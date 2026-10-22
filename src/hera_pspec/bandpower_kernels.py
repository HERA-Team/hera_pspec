"""Compact storage for bandpower window functions and covariances.

Window functions and covariances are attached to every (baseline-pair-time,
polarization-pair) entry of a ``UVPSpec``, but they rarely differ between all of
those entries: analytic window functions are identical across times, exact window
functions depend only on baseline length, and so on. Storing a full array for every
entry wastes memory and disk.

The classes here store each distinct matrix ("block") once, together with an
integer ``block_index`` of shape ``(Nbltpairs, Npols)`` that says which block
belongs to which entry::

    value(bltpair i, polpair p) = blocks[block_index[i, p]]

Classes
-------
BandpowerKernel
    Shared storage, indexing, selection, concatenation and HDF5 I/O.
BandpowerWindow
    Window functions. Rows (the first block axis) are output bandpowers; the
    remaining block axes are the "input" axes named by ``in_axes``.
BandpowerCovariance
    Bandpower covariances. Adds a per-entry ``scale`` so that a covariance that is
    a scaled copy of another one (e.g. after weighted averaging) can share its
    block::

        value(i, p) = scale[i, p] * blocks[block_index[i, p]]
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar

import attrs
import numpy as np

# Weight sums below this are treated as zero, matching grouping.average_spectra.
_MIN_WEIGHT_SUM = 1e-40


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _as_index_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.intp)


def _as_indices(selection, n: int) -> np.ndarray:
    """Convert None, an int, a slice, a boolean mask or integers to an index array."""
    if selection is None:
        return np.arange(n)
    return np.atleast_1d(np.arange(n)[selection])


def _read_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read ``dataset[rows]`` from an HDF5 dataset, reading as little as possible."""
    if rows.size == 0:
        return dataset[0:0]
    steps = np.diff(rows)
    if np.all(steps == 1):
        return dataset[rows[0] : rows[-1] + 1]
    if np.all(steps > 0):
        # h5py fancy indexing needs strictly increasing indices.
        return dataset[rows]
    return dataset[()][rows]


@attrs.define(frozen=True, eq=False, repr=False)
class BandpowerKernel:
    """Deduplicated per-entry matrices for one spectral window of a ``UVPSpec``.

    Instances are immutable; every operation returns a new object.

    Parameters
    ----------
    blocks : array_like of float
        The distinct matrices, shape ``(Nblocks, Nout, ...)``.
    block_index : array_like of int
        Shape ``(Nbltpairs, Npols)``. Entry ``(i, p)`` uses ``blocks[block_index[i, p]]``.
    """

    blocks: np.ndarray = attrs.field(converter=_as_float_array)
    block_index: np.ndarray = attrs.field(converter=_as_index_array)

    # Names of extra attributes with one value per entry, shape (Nbltpairs, Npols).
    _per_entry_fields: ClassVar[tuple[str, ...]] = ()

    @blocks.validator
    def _check_blocks(self, attribute, value):
        if value.ndim < 2:
            raise ValueError(
                f"blocks must have shape (Nblocks, Nout, ...); got {value.shape}"
            )

    @block_index.validator
    def _check_block_index(self, attribute, value):
        if value.ndim != 2:
            raise ValueError(
                f"block_index must have shape (Nbltpairs, Npols); got {value.shape}"
            )
        if value.size and (value.min() < 0 or value.max() >= self.Nblocks):
            raise ValueError(
                f"block_index values must lie in [0, {self.Nblocks}); "
                f"got range [{value.min()}, {value.max()}]"
            )

    # ------------------------------------------------------------------ shapes
    __hash__ = None

    @property
    def Nbltpairs(self) -> int:
        """Number of baseline-pair-time entries."""
        return self.block_index.shape[0]

    @property
    def Npols(self) -> int:
        """Number of polarization pairs."""
        return self.block_index.shape[1]

    @property
    def Nblocks(self) -> int:
        """Number of stored (distinct) blocks."""
        return self.blocks.shape[0]

    @property
    def block_shape(self) -> tuple[int, ...]:
        """Shape of a single block, ``(Nout, ...)``."""
        return self.blocks.shape[1:]

    @property
    def Nout(self) -> int:
        """Number of output bandpowers (rows of each block)."""
        return self.blocks.shape[1]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(Nbltpairs={self.Nbltpairs}, Npols={self.Npols}, "
            f"Nblocks={self.Nblocks}, block_shape={self.block_shape}"
            f"{self._repr_extra()})"
        )

    def _repr_extra(self) -> str:
        return ""

    # ------------------------------------------------------------ access
    def get(self, bltpair_inds, pol_ind) -> np.ndarray:
        """Return the matrices for some baseline-pair-times and one polarization pair.

        Parameters
        ----------
        bltpair_inds : int, slice, or array_like of int or bool
            Indices into the baseline-pair-time axis.
        pol_ind : int
            Index into the polarization-pair axis.

        Returns
        -------
        ndarray
            Shape ``(N, *block_shape)``, or ``block_shape`` for a scalar index.
        """
        return self.blocks[self.block_index[bltpair_inds, pol_ind]]

    def _entries(self) -> np.ndarray:
        """All entries, shape ``(Nbltpairs, Npols, *block_shape)``."""
        return self.blocks[self.block_index]

    def to_dense(self) -> np.ndarray:
        """Expand to the legacy dense layout ``(Nbltpairs, *block_shape, Npols)``.

        This allocates the full array, which is what this class exists to avoid; use it
        for compatibility and tests only.
        """
        return np.moveaxis(self._entries(), 1, -1)

    @classmethod
    def from_dense(cls, array, deduplicate: bool = True, **kwargs):
        """Build from the legacy dense layout ``(Nbltpairs, *block_shape, Npols)``.

        Parameters
        ----------
        array : array_like
            Dense array with the polarization-pair axis last.
        deduplicate : bool
            Merge entries whose matrices are exactly equal. Default True.
        **kwargs
            Extra attributes for the subclass (e.g. ``in_axes`` for a window).
        """
        entries = np.moveaxis(_as_float_array(array), -1, 1)
        nbltpairs, npols = entries.shape[:2]
        kernel = cls(
            blocks=entries.reshape(nbltpairs * npols, *entries.shape[2:]),
            block_index=np.arange(nbltpairs * npols).reshape(nbltpairs, npols),
            **kwargs,
        )
        return kernel.deduplicate() if deduplicate else kernel

    # --------------------------------------------------- structural operations
    def deduplicate(self):
        """Merge exactly equal blocks and drop blocks that no entry uses."""
        used, entry_to_used = np.unique(self.block_index, return_inverse=True)
        unique_blocks, used_to_unique = np.unique(
            self.blocks[used].reshape(len(used), -1), axis=0, return_inverse=True
        )
        block_index = used_to_unique.reshape(-1)[entry_to_used.reshape(-1)]
        return attrs.evolve(
            self,
            blocks=unique_blocks.reshape(-1, *self.block_shape),
            block_index=block_index.reshape(self.block_index.shape),
        )

    def select(self, bltpair_inds=None, pol_inds=None):
        """Return the kernel for a subset of entries, dropping unused blocks.

        Parameters
        ----------
        bltpair_inds, pol_inds : None, int, slice, or array_like of int or bool
            Entries to keep along each axis. None keeps all of them.
        """
        bltpairs = _as_indices(bltpair_inds, self.Nbltpairs)
        pols = _as_indices(pol_inds, self.Npols)
        index = self.block_index[np.ix_(bltpairs, pols)]
        used, inverse = np.unique(index, return_inverse=True)
        return attrs.evolve(
            self,
            blocks=self.blocks[used],
            block_index=inverse.reshape(index.shape),
            **self._select_extra(bltpairs, pols),
        )

    def _select_extra(self, bltpairs: np.ndarray, pols: np.ndarray) -> dict:
        return {
            name: getattr(self, name)[np.ix_(bltpairs, pols)]
            for name in self._per_entry_fields
        }

    @classmethod
    def concatenate(cls, kernels: Sequence[BandpowerKernel], axis: str = "bltpair"):
        """Join kernels along the baseline-pair-time or polarization-pair axis.

        Blocks are not deduplicated across inputs; call :meth:`deduplicate` on the
        result if they may share matrices.

        Parameters
        ----------
        kernels : sequence of BandpowerKernel
            Kernels of this class with the same ``block_shape``.
        axis : {"bltpair", "pol"}
            Axis to join along.
        """
        kernels = list(kernels)
        if not kernels:
            raise ValueError("need at least one kernel to concatenate")
        if axis not in ("bltpair", "pol"):
            raise ValueError(f"axis must be 'bltpair' or 'pol'; got {axis!r}")
        for kernel in kernels:
            if type(kernel) is not cls:
                raise TypeError(
                    f"can only concatenate {cls.__name__} objects; "
                    f"got {type(kernel).__name__}"
                )
            if kernel.block_shape != kernels[0].block_shape:
                raise ValueError(
                    "cannot concatenate kernels with block shapes "
                    f"{kernels[0].block_shape} and {kernel.block_shape}"
                )
        offsets = np.cumsum([0] + [kernel.Nblocks for kernel in kernels[:-1]])
        array_axis = 0 if axis == "bltpair" else 1
        return attrs.evolve(
            kernels[0],
            blocks=np.concatenate([kernel.blocks for kernel in kernels]),
            block_index=np.concatenate(
                [k.block_index + off for k, off in zip(kernels, offsets)],
                axis=array_axis,
            ),
            **cls._concatenate_extra(kernels, axis),
        )

    @classmethod
    def _concatenate_extra(cls, kernels: list, axis: str) -> dict:
        array_axis = 0 if axis == "bltpair" else 1
        return {
            name: np.concatenate(
                [getattr(kernel, name) for kernel in kernels], axis=array_axis
            )
            for name in cls._per_entry_fields
        }

    # ------------------------------------------------------------ comparison
    def __eq__(self, other) -> bool:
        """Equal when every entry holds the same matrix, however the blocks are laid out."""
        if type(other) is not type(self):
            return NotImplemented
        if (
            self.block_index.shape != other.block_index.shape
            or self.block_shape != other.block_shape
            or not self._extra_equal(other)
        ):
            return False
        # Compare each distinct pairing of entries once.
        keys = self._entry_keys()
        width = keys.shape[-1]
        pairs = np.unique(
            np.concatenate(
                [keys.reshape(-1, width), other._entry_keys().reshape(-1, width)],
                axis=1,
            ),
            axis=0,
        )
        return all(
            np.allclose(
                self._block_from_key(row[:width]), other._block_from_key(row[width:])
            )
            for row in pairs
        )

    def _extra_equal(self, other) -> bool:
        return True

    def _entry_keys(self) -> np.ndarray:
        """Per-entry values that determine the matrix, shape ``(Nbltpairs, Npols, k)``."""
        return self.block_index[..., None].astype(np.float64)

    def _block_from_key(self, key: np.ndarray) -> np.ndarray:
        return self.blocks[int(key[0])]

    # ----------------------------------------------------------------- HDF5
    def write_hdf5(self, group) -> None:
        """Write to an (empty) ``h5py.Group``."""
        group.attrs["kernel_type"] = type(self).__name__
        group.create_dataset("blocks", data=self.blocks, dtype=np.float64)
        group.create_dataset("block_index", data=self.block_index, dtype=np.int64)
        for name in self._per_entry_fields:
            group.create_dataset(name, data=getattr(self, name), dtype=np.float64)
        self._write_extra(group)

    def _write_extra(self, group) -> None:
        pass

    @classmethod
    def read_hdf5(cls, group, bltpair_inds=None, pol_inds=None):
        """Read from an ``h5py.Group`` written by :meth:`write_hdf5`.

        Only the selected entries, and the blocks they use, are read from disk.

        Parameters
        ----------
        group : h5py.Group
            Group holding the kernel.
        bltpair_inds, pol_inds : None, int, slice, or array_like of int or bool
            Entries to read along each axis. None reads all of them.
        """
        kernel_type = group.attrs.get("kernel_type")
        if kernel_type != cls.__name__:
            raise ValueError(
                f"HDF5 group holds a {kernel_type!r}, not a {cls.__name__!r}"
            )
        nbltpairs, npols = group["block_index"].shape
        bltpairs = _as_indices(bltpair_inds, nbltpairs)
        pols = _as_indices(pol_inds, npols)

        index = _read_rows(group["block_index"], bltpairs)[:, pols]
        used, inverse = np.unique(index, return_inverse=True)
        blocks = group["blocks"]
        if len(used) == blocks.shape[0]:
            blocks = blocks[()]
        else:
            blocks = _read_rows(blocks, used)

        extra = {
            name: _read_rows(group[name], bltpairs)[:, pols]
            for name in cls._per_entry_fields
        }
        extra.update(cls._read_extra(group, pols))
        return cls(blocks=blocks, block_index=inverse.reshape(index.shape), **extra)

    @classmethod
    def _read_extra(cls, group, pols: np.ndarray) -> dict:
        return {}


def _check_in_coords(instance, attribute, value):
    for name, coords in value.items():
        if name not in instance.in_axes:
            raise ValueError(
                f"in_coords has an entry for {name!r}, which is not in in_axes "
                f"{instance.in_axes}"
            )
        axis_length = instance.block_shape[1 + instance.in_axes.index(name)]
        if coords.shape != (instance.Npols, axis_length):
            raise ValueError(
                f"in_coords[{name!r}] must have shape (Npols, {axis_length}) = "
                f"{(instance.Npols, axis_length)}; got {coords.shape}"
            )


@attrs.define(frozen=True, eq=False, repr=False)
class BandpowerWindow(BandpowerKernel):
    """Bandpower window functions for one spectral window.

    Each block has shape ``(Nout, *in_shape)``: row ``i`` is the window function of
    output bandpower ``i`` over the input axes.

    Parameters
    ----------
    blocks, block_index
        See :class:`BandpowerKernel`.
    in_axes : tuple of str
        Names of the input axes, one per block axis after the first. Defaults to
        ``("delay",)`` for square delay-delay windows; exact cylindrical windows use
        ``("kperp", "kpara")``.
    in_coords : dict of str to array_like
        Optional coordinates (e.g. bin centres) for named input axes, each of shape
        ``(Npols, axis_length)`` because grids can differ between polarization pairs.
    """

    in_axes: tuple[str, ...] = attrs.field(converter=tuple, default=("delay",))
    in_coords: dict[str, np.ndarray] = attrs.field(
        factory=dict,
        converter=lambda d: {str(k): _as_float_array(v) for k, v in d.items()},
        validator=_check_in_coords,
    )

    @in_axes.validator
    def _check_in_axes(self, attribute, value):
        if len(value) != len(self.block_shape) - 1:
            raise ValueError(
                f"in_axes {value} names {len(value)} input axes but blocks have "
                f"{len(self.block_shape) - 1} (block_shape {self.block_shape})"
            )

    def _repr_extra(self) -> str:
        return f", in_axes={self.in_axes}"

    def _select_extra(self, bltpairs, pols) -> dict:
        extra = super()._select_extra(bltpairs, pols)
        extra["in_coords"] = {k: v[pols] for k, v in self.in_coords.items()}
        return extra

    @classmethod
    def _concatenate_extra(cls, kernels, axis) -> dict:
        extra = super()._concatenate_extra(kernels, axis)
        first = kernels[0]
        for kernel in kernels[1:]:
            if kernel.in_axes != first.in_axes:
                raise ValueError(
                    f"cannot concatenate windows with in_axes {first.in_axes} "
                    f"and {kernel.in_axes}"
                )
            if kernel.in_coords.keys() != first.in_coords.keys():
                raise ValueError("cannot concatenate windows with different in_coords")
        if axis == "pol":
            extra["in_coords"] = {
                name: np.concatenate([kernel.in_coords[name] for kernel in kernels])
                for name in first.in_coords
            }
        else:
            for kernel in kernels[1:]:
                for name, coords in first.in_coords.items():
                    if not np.array_equal(kernel.in_coords[name], coords):
                        raise ValueError(
                            f"cannot concatenate windows whose in_coords[{name!r}] "
                            "differ"
                        )
        return extra

    def _extra_equal(self, other) -> bool:
        return (
            self.in_axes == other.in_axes
            and self.in_coords.keys() == other.in_coords.keys()
            and all(
                np.allclose(self.in_coords[k], other.in_coords[k])
                for k in self.in_coords
            )
        )

    def _write_extra(self, group) -> None:
        group.attrs["in_axes"] = json.dumps(list(self.in_axes))
        coords = group.create_group("in_coords")
        for name, values in self.in_coords.items():
            coords.create_dataset(name, data=values, dtype=np.float64)

    @classmethod
    def _read_extra(cls, group, pols) -> dict:
        return {
            "in_axes": tuple(json.loads(group.attrs["in_axes"])),
            "in_coords": {
                name: dataset[()][pols] for name, dataset in group["in_coords"].items()
            },
        }

    def average(self, groups: Sequence, weights=None):
        """Weighted average of window functions over groups of baseline-pair-times.

        Output entry ``(g, p)`` has rows
        ``W_out[d] = sum_k w_k[d] W_k[d] / sum_k w_k[d]`` over the members ``k`` of
        group ``g``, the same normalisation ``grouping.average_spectra`` uses. When
        every member shares one block and no row has zero total weight, that block is
        reused unchanged, so averaging adds no storage.

        Parameters
        ----------
        groups : sequence of index arrays
            Each item selects the baseline-pair-times averaged into one output entry.
        weights : array_like, optional
            Shape ``(Nbltpairs, Npols, Nout)``: a weight per entry and output
            bandpower. Default: equal weights.

        Returns
        -------
        BandpowerWindow
            With ``Nbltpairs == len(groups)``.
        """
        weights = _check_weights(weights, self)
        new_blocks: list[np.ndarray] = []
        block_index = np.empty((len(groups), self.Npols), dtype=np.intp)
        row_shape = (self.Nout,) + (1,) * (len(self.block_shape) - 1)
        for g, members in enumerate(groups):
            members = _as_group(members, self.Nbltpairs)
            for p in range(self.Npols):
                ids = self.block_index[members, p]
                w = weights[members, p]
                wsum = w.sum(axis=0)
                if np.all(ids == ids[0]) and np.all(wsum > _MIN_WEIGHT_SUM):
                    block_index[g, p] = ids[0]
                    continue
                averaged = np.einsum("kd,kd...->d...", w, self.blocks[ids])
                averaged /= wsum.clip(_MIN_WEIGHT_SUM, np.inf).reshape(row_shape)
                block_index[g, p] = self.Nblocks + len(new_blocks)
                new_blocks.append(averaged)
        return _with_new_blocks(self, new_blocks, block_index)


@attrs.define(frozen=True, eq=False, repr=False)
class BandpowerCovariance(BandpowerKernel):
    """Bandpower covariances for one spectral window.

    Each block is a square ``(Nout, Nout)`` matrix, and entry ``(i, p)`` is
    ``scale[i, p] * blocks[block_index[i, p]]``. The scale lets weighted averages of
    entries that share a block keep sharing it.

    Real and imaginary parts of the power spectrum have separate covariances; store
    them as two instances.

    Parameters
    ----------
    blocks, block_index
        See :class:`BandpowerKernel`.
    scale : array_like of float, optional
        Shape ``(Nbltpairs, Npols)``. Default: all ones.
    """

    scale: np.ndarray = attrs.field(converter=_as_float_array)

    _per_entry_fields: ClassVar[tuple[str, ...]] = ("scale",)

    @scale.default
    def _default_scale(self):
        return np.ones(np.shape(self.block_index))

    @scale.validator
    def _check_scale(self, attribute, value):
        if value.shape != self.block_index.shape:
            raise ValueError(
                f"scale must have the shape of block_index {self.block_index.shape}; "
                f"got {value.shape}"
            )

    def __attrs_post_init__(self):
        if len(self.block_shape) != 2 or self.block_shape[0] != self.block_shape[1]:
            raise ValueError(
                f"covariance blocks must be square (Nout, Nout); got {self.block_shape}"
            )

    def get(self, bltpair_inds, pol_ind) -> np.ndarray:
        """Return the covariance matrices for some entries; see :meth:`BandpowerKernel.get`."""
        scale = np.asarray(self.scale[bltpair_inds, pol_ind])
        return scale[..., None, None] * super().get(bltpair_inds, pol_ind)

    def _entries(self) -> np.ndarray:
        return self.scale[..., None, None] * super()._entries()

    def _entry_keys(self) -> np.ndarray:
        return np.stack([self.block_index.astype(np.float64), self.scale], axis=-1)

    def _block_from_key(self, key: np.ndarray) -> np.ndarray:
        return key[1] * self.blocks[int(key[0])]

    def average(self, groups: Sequence, weights=None):
        """Weighted average of covariances over groups of baseline-pair-times.

        Output entry ``(g, p)`` is
        ``C_out[d, e] = sum_k w_k[d] w_k[e] C_k[d, e] / (W[d] W[e])`` with
        ``W = sum_k w_k``, the covariance of the weighted mean that
        ``grouping.average_spectra`` computes. When the members share one block and
        each member's weights are the same for every bandpower (e.g. integration-time
        weights), the result is a rescaled copy of that block and only ``scale``
        changes.

        Parameters
        ----------
        groups : sequence of index arrays
            Each item selects the baseline-pair-times averaged into one output entry.
        weights : array_like, optional
            Shape ``(Nbltpairs, Npols, Nout)``. Default: equal weights.

        Returns
        -------
        BandpowerCovariance
            With ``Nbltpairs == len(groups)``.
        """
        weights = _check_weights(weights, self)
        new_blocks: list[np.ndarray] = []
        block_index = np.empty((len(groups), self.Npols), dtype=np.intp)
        scale = np.ones((len(groups), self.Npols))
        for g, members in enumerate(groups):
            members = _as_group(members, self.Nbltpairs)
            for p in range(self.Npols):
                ids = self.block_index[members, p]
                s = self.scale[members, p]
                w = weights[members, p]
                wsum = w.sum(axis=0).clip(_MIN_WEIGHT_SUM, np.inf)
                if np.all(ids == ids[0]) and np.all(w == w[:, :1]):
                    # C_out = sum_k a_k^2 s_k B / (sum_k a_k)^2
                    a = w[:, 0]
                    block_index[g, p] = ids[0]
                    scale[g, p] = np.sum(a**2 * s) / wsum[0] ** 2
                    continue
                averaged = np.einsum(
                    "kd,ke,k,kde->de", w, w, s, self.blocks[ids]
                ) / np.outer(wsum, wsum)
                block_index[g, p] = self.Nblocks + len(new_blocks)
                new_blocks.append(averaged)
        return _with_new_blocks(self, new_blocks, block_index, scale=scale)


def _check_weights(weights, kernel: BandpowerKernel) -> np.ndarray:
    shape = (kernel.Nbltpairs, kernel.Npols, kernel.Nout)
    if weights is None:
        return np.ones(shape)
    weights = _as_float_array(weights)
    if weights.shape != shape:
        raise ValueError(
            f"weights must have shape (Nbltpairs, Npols, Nout) = {shape}; "
            f"got {weights.shape}"
        )
    return weights


def _as_group(members, n: int) -> np.ndarray:
    members = _as_indices(members, n)
    if members.size == 0:
        raise ValueError("cannot average an empty group")
    return members


def _with_new_blocks(kernel, new_blocks, block_index, **extra):
    """Append ``new_blocks`` to ``kernel.blocks``, then drop blocks nothing uses."""
    blocks = kernel.blocks
    if new_blocks:
        blocks = np.concatenate([blocks, np.stack(new_blocks)])
    return attrs.evolve(
        kernel, blocks=blocks, block_index=block_index, **extra
    ).select()
