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

Averaging these objects lives in :mod:`hera_pspec.grouping`
(``average_bandpower_window`` and ``average_bandpower_covariance``).

Examples
--------
Analytic window functions are computed once per baseline pair and polarization
pair and are the same at every time. Converting the dense array a ``UVPSpec``
currently stores keeps one block per (baseline pair, polarization pair)::

    dense = uvp.window_function_array[spw]   # (Nbltpairs, Ndly, Ndly, Npols)
    window = BandpowerWindow.from_dense(dense)
    window.Nblocks                           # Nblpairs * Npols, not Nbltpairs * Npols
    window.get(bltpair_inds, pol_ind)        # (N, Ndly, Ndly), as dense[bltpair_inds, ..., pol_ind]

Exact window functions are one block per baseline-length group and polarization
pair, with the (kperp, kpara) grid attached::

    window = BandpowerWindow(
        blocks=wf_per_group,                     # (Ngroups * Npols, Ndly, Nkperp, Nkpara)
        block_index=group_of_each_entry,         # (Nbltpairs, Npols)
        in_axes=("kperp", "kpara"),
        in_coords={"kperp": kperp_bins, "kpara": kpara_bins},  # each (Npols, n)
    )

Selecting baseline-pair-times, or joining two objects (as ``combine_uvpspec`` and
``UVPSpec.select`` need to), keeps the sharing::

    window.select(bltpair_inds=[0, 5, 9], pol_inds=[0])
    BandpowerWindow.concatenate([window_a, window_b], axis="bltpair")
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence

import attrs
import numpy as np


def _as_indices(selection, n: int) -> np.ndarray:
    """Turn a selection along an axis of length ``n`` into an array of indices.

    ``selection`` may be None (everything), an int, a slice, a boolean mask or a
    sequence of ints. The result is always 1D, so the selected axis is kept even
    for a single int.
    """
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

    blocks: np.ndarray = attrs.field(
        converter=functools.partial(np.asarray, dtype=np.float64)
    )
    block_index: np.ndarray = attrs.field(
        converter=functools.partial(np.asarray, dtype=np.intp)
    )

    __hash__ = None

    @blocks.validator
    def _check_blocks(self, attribute, value):
        """Blocks need an axis for blocks and at least one for output bandpowers."""
        if value.ndim < 2:
            raise ValueError(
                f"blocks must have shape (Nblocks, Nout, ...); got {value.shape}"
            )

    @block_index.validator
    def _check_block_index(self, attribute, value):
        """block_index must be 2D and point at existing blocks."""
        if value.ndim != 2:
            raise ValueError(
                f"block_index must have shape (Nbltpairs, Npols); got {value.shape}"
            )
        if value.size and (value.min() < 0 or value.max() >= self.Nblocks):
            raise ValueError(
                f"block_index values must lie in [0, {self.Nblocks}); "
                f"got range [{value.min()}, {value.max()}]"
            )

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
            f"Nblocks={self.Nblocks}, block_shape={self.block_shape})"
        )

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

    def to_dense(self) -> np.ndarray:
        """Expand to the legacy dense layout ``(Nbltpairs, *block_shape, Npols)``.

        This allocates the full array, which is what this class exists to avoid; use it
        for compatibility and tests only.
        """
        return np.moveaxis(self.blocks[self.block_index], 1, -1)

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
            Other attributes of the class being built, e.g. ``in_axes`` for a
            ``BandpowerWindow``.
        """
        entries = np.moveaxis(np.asarray(array, dtype=np.float64), -1, 1)
        nbltpairs, npols = entries.shape[:2]
        kernel = cls(
            blocks=entries.reshape(nbltpairs * npols, *entries.shape[2:]),
            block_index=np.arange(nbltpairs * npols).reshape(nbltpairs, npols),
            **kwargs,
        )
        return kernel.deduplicate() if deduplicate else kernel

    def deduplicate(self):
        """Merge exactly equal blocks and drop blocks that no entry uses.

        Blocks that are only proportional to each other are not merged.
        """
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
        return self._take(
            _as_indices(bltpair_inds, self.Nbltpairs), _as_indices(pol_inds, self.Npols)
        )

    def _take(self, bltpairs: np.ndarray, pols: np.ndarray, **attributes):
        """Copy holding only entries ``bltpairs`` x ``pols``, with unused blocks dropped.

        Subclasses' ``select`` pass their own attributes, already cut down to the
        selected entries, as keyword arguments.
        """
        index = self.block_index[np.ix_(bltpairs, pols)]
        used, inverse = np.unique(index, return_inverse=True)
        return attrs.evolve(
            self,
            blocks=self.blocks[used],
            block_index=inverse.reshape(index.shape),
            **attributes,
        )

    @classmethod
    def concatenate(cls, kernels: Sequence[BandpowerKernel], axis: str = "bltpair"):
        """Join kernels along the baseline-pair-time or polarization-pair axis.

        Needed to combine power spectra (``combine_uvpspec``). Blocks are not
        deduplicated across inputs; call :meth:`deduplicate` on the result if they may
        share matrices.

        Parameters
        ----------
        kernels : sequence of BandpowerKernel
            Kernels of this class with the same ``block_shape``.
        axis : {"bltpair", "pol"}
            Axis to join along.
        """
        return cls._join(cls._check_joinable(kernels, axis), axis)

    @classmethod
    def _check_joinable(cls, kernels: Sequence[BandpowerKernel], axis: str) -> list:
        """Raise unless ``kernels`` are one or more objects of this class with equal
        block shapes and ``axis`` is valid; return them as a list.
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
        return kernels

    @classmethod
    def _join(cls, kernels: list, axis: str, **attributes):
        """Join the blocks and indices of kernels that passed :meth:`_check_joinable`.

        Subclasses' ``concatenate`` pass their own attributes, already joined, as
        keyword arguments.
        """
        # Each kernel's indices point into its own blocks; shift them past the
        # blocks of the kernels before it.
        offsets = np.cumsum([0] + [kernel.Nblocks for kernel in kernels[:-1]])
        return attrs.evolve(
            kernels[0],
            blocks=np.concatenate([kernel.blocks for kernel in kernels]),
            block_index=np.concatenate(
                [k.block_index + off for k, off in zip(kernels, offsets)],
                axis=0 if axis == "bltpair" else 1,
            ),
            **attributes,
        )

    def __eq__(self, other) -> bool:
        """Equal when every entry holds the same matrix, however the blocks are laid out."""
        if type(other) is not type(self):
            return NotImplemented
        if (
            self.block_index.shape != other.block_index.shape
            or self.block_shape != other.block_shape
        ):
            return False
        # Compare each distinct (own block, other block) pairing once.
        pairs = np.unique(
            np.stack([self.block_index.ravel(), other.block_index.ravel()], axis=1),
            axis=0,
        )
        return all(np.allclose(self.blocks[a], other.blocks[b]) for a, b in pairs)

    def write_hdf5(self, group) -> None:
        """Write to an empty ``h5py.Group``."""
        if len(group) or "kernel_type" in group.attrs:
            raise ValueError(
                f"HDF5 group {group.name!r} is not empty; write to a new group"
            )
        group.attrs["kernel_type"] = type(self).__name__
        group.create_dataset("blocks", data=self.blocks, dtype=np.float64)
        group.create_dataset("block_index", data=self.block_index, dtype=np.int64)

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
        _, _, blocks, block_index = cls._read_blocks(group, bltpair_inds, pol_inds)
        return cls(blocks=blocks, block_index=block_index)

    @classmethod
    def _read_blocks(cls, group, bltpair_inds, pol_inds):
        """Read the selected part of ``block_index`` and the blocks it uses.

        Returns
        -------
        bltpairs, pols : ndarray
            The selections as index arrays, for subclasses to read their attributes.
        blocks, block_index : ndarray
            Arguments for the constructor.
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
        # When the selection uses every block, one contiguous read is cheapest;
        # otherwise read only the blocks it uses.
        if len(used) == blocks.shape[0]:
            blocks = blocks[()]
        else:
            blocks = _read_rows(blocks, used)
        return bltpairs, pols, blocks, inverse.reshape(index.shape)


def _check_in_coords(instance, attribute, value):
    """Each coordinate array must belong to a named input axis and match its length."""
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
        converter=lambda d: {
            str(k): np.asarray(v, dtype=np.float64) for k, v in d.items()
        },
        validator=_check_in_coords,
    )

    @in_axes.validator
    def _check_in_axes(self, attribute, value):
        """There must be one name per block axis after the output axis."""
        if len(value) != len(self.block_shape) - 1:
            raise ValueError(
                f"in_axes {value} names {len(value)} input axes but blocks have "
                f"{len(self.block_shape) - 1} (block_shape {self.block_shape})"
            )

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, in_axes={self.in_axes})"

    def select(self, bltpair_inds=None, pol_inds=None):
        """Return the window for a subset of entries; see :meth:`BandpowerKernel.select`."""
        bltpairs = _as_indices(bltpair_inds, self.Nbltpairs)
        pols = _as_indices(pol_inds, self.Npols)
        return self._take(
            bltpairs,
            pols,
            in_coords={name: coords[pols] for name, coords in self.in_coords.items()},
        )

    @classmethod
    def concatenate(cls, kernels: Sequence[BandpowerWindow], axis: str = "bltpair"):
        """Join windows along an axis; see :meth:`BandpowerKernel.concatenate`.

        All windows need the same ``in_axes`` and ``in_coords`` names. Along
        ``"bltpair"`` the coordinates must also be equal; along ``"pol"`` they are
        joined.
        """
        kernels = cls._check_joinable(kernels, axis)
        first = kernels[0]
        for kernel in kernels[1:]:
            if kernel.in_axes != first.in_axes:
                raise ValueError(
                    f"cannot concatenate windows with in_axes {first.in_axes} "
                    f"and {kernel.in_axes}"
                )
            if kernel.in_coords.keys() != first.in_coords.keys():
                raise ValueError("cannot concatenate windows with different in_coords")
            if axis == "bltpair" and not all(
                np.array_equal(kernel.in_coords[name], coords)
                for name, coords in first.in_coords.items()
            ):
                raise ValueError(
                    "cannot concatenate along 'bltpair' windows whose in_coords differ"
                )
        if axis == "pol":
            in_coords = {
                name: np.concatenate([kernel.in_coords[name] for kernel in kernels])
                for name in first.in_coords
            }
        else:
            in_coords = first.in_coords
        return cls._join(kernels, axis, in_coords=in_coords)

    def __eq__(self, other) -> bool:
        """Equal when ``in_axes``, ``in_coords`` and every entry's matrix are equal."""
        if type(other) is not type(self):
            return NotImplemented
        return (
            self.in_axes == other.in_axes
            and self.in_coords.keys() == other.in_coords.keys()
            and all(
                np.allclose(coords, other.in_coords[name])
                for name, coords in self.in_coords.items()
            )
            and super().__eq__(other)
        )

    def write_hdf5(self, group) -> None:
        """Write to an empty ``h5py.Group``."""
        super().write_hdf5(group)
        group.attrs["in_axes"] = json.dumps(list(self.in_axes))
        coords = group.create_group("in_coords")
        for name, values in self.in_coords.items():
            coords.create_dataset(name, data=values, dtype=np.float64)

    @classmethod
    def read_hdf5(cls, group, bltpair_inds=None, pol_inds=None):
        """Read from an ``h5py.Group``; see :meth:`BandpowerKernel.read_hdf5`."""
        _, pols, blocks, block_index = cls._read_blocks(group, bltpair_inds, pol_inds)
        return cls(
            blocks=blocks,
            block_index=block_index,
            in_axes=tuple(json.loads(group.attrs["in_axes"])),
            in_coords={
                name: dataset[()][pols] for name, dataset in group["in_coords"].items()
            },
        )


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

    scale: np.ndarray = attrs.field(
        converter=functools.partial(np.asarray, dtype=np.float64)
    )

    @scale.default
    def _default_scale(self):
        """All ones: every entry is its block, unscaled."""
        return np.ones(np.shape(self.block_index))

    @scale.validator
    def _check_scale(self, attribute, value):
        """There is one scale per entry."""
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
        """Return the scaled covariance matrices; see :meth:`BandpowerKernel.get`."""
        scale = np.asarray(self.scale[bltpair_inds, pol_ind])
        return scale[..., None, None] * super().get(bltpair_inds, pol_ind)

    def to_dense(self) -> np.ndarray:
        """Expand to the legacy dense layout ``(Nbltpairs, Nout, Nout, Npols)``."""
        return np.moveaxis(
            self.scale[..., None, None] * self.blocks[self.block_index], 1, -1
        )

    def select(self, bltpair_inds=None, pol_inds=None):
        """Return the covariance for a subset of entries; see :meth:`BandpowerKernel.select`."""
        bltpairs = _as_indices(bltpair_inds, self.Nbltpairs)
        pols = _as_indices(pol_inds, self.Npols)
        return self._take(bltpairs, pols, scale=self.scale[np.ix_(bltpairs, pols)])

    @classmethod
    def concatenate(cls, kernels: Sequence[BandpowerCovariance], axis: str = "bltpair"):
        """Join covariances along an axis; see :meth:`BandpowerKernel.concatenate`."""
        kernels = cls._check_joinable(kernels, axis)
        scale = np.concatenate(
            [kernel.scale for kernel in kernels], axis=0 if axis == "bltpair" else 1
        )
        return cls._join(kernels, axis, scale=scale)

    def __eq__(self, other) -> bool:
        """Equal when every entry's scaled matrix is equal, however it is stored."""
        if type(other) is not type(self):
            return NotImplemented
        if (
            self.block_index.shape != other.block_index.shape
            or self.block_shape != other.block_shape
        ):
            return False
        # Compare each distinct (own block, own scale, other block, other scale) once.
        rows = np.unique(
            np.stack(
                [
                    self.block_index.ravel(),
                    self.scale.ravel(),
                    other.block_index.ravel(),
                    other.scale.ravel(),
                ],
                axis=1,
            ),
            axis=0,
        )
        return all(
            np.allclose(s * self.blocks[int(a)], t * other.blocks[int(b)])
            for a, s, b, t in rows
        )

    def write_hdf5(self, group) -> None:
        """Write to an empty ``h5py.Group``."""
        super().write_hdf5(group)
        group.create_dataset("scale", data=self.scale, dtype=np.float64)

    @classmethod
    def read_hdf5(cls, group, bltpair_inds=None, pol_inds=None):
        """Read from an ``h5py.Group``; see :meth:`BandpowerKernel.read_hdf5`."""
        bltpairs, pols, blocks, block_index = cls._read_blocks(
            group, bltpair_inds, pol_inds
        )
        return cls(
            blocks=blocks,
            block_index=block_index,
            scale=_read_rows(group["scale"], bltpairs)[:, pols],
        )
