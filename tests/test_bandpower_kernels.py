import h5py
import numpy as np
import pytest

from hera_pspec import BandpowerCovariance, BandpowerKernel, BandpowerWindow

NBLPAIRS, NTIMES, NPOLS, NDLYS = 3, 4, 2, 5
NBLTPAIRS = NBLPAIRS * NTIMES


def repeat_over_times(per_blpair: np.ndarray) -> np.ndarray:
    """Dense (Nbltpairs, ..., Npols) array from (Nblpairs, Npols, ...) blocks, time-major."""
    dense = np.tile(per_blpair, (NTIMES,) + (1,) * (per_blpair.ndim - 1))
    return np.moveaxis(dense, 1, -1)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def window_dense(rng) -> np.ndarray:
    """Analytic-style windows: one matrix per (blpair, pol), repeated over times."""
    return repeat_over_times(rng.random((NBLPAIRS, NPOLS, NDLYS, NDLYS)))


@pytest.fixture()
def cov_dense(rng) -> np.ndarray:
    """Symmetric covariances: one matrix per (blpair, pol), repeated over times."""
    blocks = rng.random((NBLPAIRS, NPOLS, NDLYS, NDLYS))
    return repeat_over_times(blocks @ np.swapaxes(blocks, -1, -2))


@pytest.fixture()
def scaled_cov(cov_dense) -> BandpowerCovariance:
    """Covariances whose scale differs at every entry."""
    cov = BandpowerCovariance.from_dense(cov_dense)
    scale = np.arange(1, NBLTPAIRS * NPOLS + 1, dtype=float).reshape(NBLTPAIRS, NPOLS)
    return BandpowerCovariance(
        blocks=cov.blocks, block_index=cov.block_index, scale=scale
    )


@pytest.fixture()
def exact_window(rng) -> BandpowerWindow:
    """Exact-style windows: one (Ndly, Nkperp, Nkpara) block per pol, shared by all bltpairs."""
    nkperp, nkpara = 3, 6
    return BandpowerWindow(
        blocks=rng.random((NPOLS, NDLYS, nkperp, nkpara)),
        block_index=np.tile(np.arange(NPOLS), (NBLTPAIRS, 1)),
        in_axes=("kperp", "kpara"),
        in_coords={
            "kperp": rng.random((NPOLS, nkperp)),
            "kpara": rng.random((NPOLS, nkpara)),
        },
    )


class TestConstruction:
    def test_from_dense_deduplicates_repeated_times(self, window_dense):
        """A window repeated over times keeps one block per (blpair, pol)."""
        window = BandpowerWindow.from_dense(window_dense)
        assert window.Nbltpairs == NBLTPAIRS
        assert window.Npols == NPOLS
        assert window.block_shape == (NDLYS, NDLYS)
        assert window.Nout == NDLYS
        assert window.blocks.shape == (NBLPAIRS * NPOLS, NDLYS, NDLYS)
        # Entries are time-major: rows t * NBLPAIRS + b. Every time of a blpair must
        # point at the same block, and different (blpair, pol) at different blocks.
        index = window.block_index.reshape(NTIMES, NBLPAIRS, NPOLS)
        np.testing.assert_array_equal(index, np.broadcast_to(index[0], index.shape))
        assert len(np.unique(index[0])) == NBLPAIRS * NPOLS
        np.testing.assert_array_equal(window.to_dense(), window_dense)

    def test_from_dense_without_deduplication(self, window_dense):
        """deduplicate=False keeps one block per entry but represents the same windows."""
        window = BandpowerWindow.from_dense(window_dense, deduplicate=False)
        assert window.Nblocks == NBLTPAIRS * NPOLS
        np.testing.assert_array_equal(
            window.block_index, np.arange(NBLTPAIRS * NPOLS).reshape(NBLTPAIRS, NPOLS)
        )
        assert window == BandpowerWindow.from_dense(window_dense)

    def test_from_dense_exact_window(self, exact_window):
        """Dense exact windows round-trip, collapsing to one block per pol."""
        dense = exact_window.to_dense()
        assert dense.shape == (NBLTPAIRS, NDLYS, 3, 6, NPOLS)
        rebuilt = BandpowerWindow.from_dense(
            dense, in_axes=exact_window.in_axes, in_coords=exact_window.in_coords
        )
        assert rebuilt.Nblocks == NPOLS
        assert rebuilt == exact_window

    def test_covariance_default_scale(self, cov_dense):
        """Without a scale, every covariance entry is its block unscaled."""
        cov = BandpowerCovariance.from_dense(cov_dense)
        np.testing.assert_array_equal(cov.scale, np.ones((NBLTPAIRS, NPOLS)))
        np.testing.assert_allclose(cov.to_dense(), cov_dense)

    def test_repr(self, exact_window, cov_dense):
        """repr shows sizes instead of dumping the arrays."""
        assert repr(BandpowerCovariance.from_dense(cov_dense)) == (
            "BandpowerCovariance(Nbltpairs=12, Npols=2, Nblocks=6, block_shape=(5, 5))"
        )
        assert repr(exact_window) == (
            "BandpowerWindow(Nbltpairs=12, Npols=2, Nblocks=2, "
            "block_shape=(5, 3, 6), in_axes=('kperp', 'kpara'))"
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"blocks": np.ones(3), "block_index": [[0]]}, "blocks must have shape"),
            ({"blocks": np.ones((1, 2, 2)), "block_index": [0]}, "block_index must"),
            ({"blocks": np.ones((1, 2, 2)), "block_index": [[1]]}, r"lie in \[0, 1\)"),
            ({"blocks": np.ones((1, 2, 2)), "block_index": [[-1]]}, r"lie in \[0, 1\)"),
        ],
    )
    def test_base_validation(self, kwargs, match):
        """Bad block or index shapes, and indices pointing past the blocks, raise."""
        with pytest.raises(ValueError, match=match):
            BandpowerKernel(**kwargs)

    def test_window_in_axes_must_match_block_ndim(self):
        """A 3D block needs two input-axis names, not the one default."""
        with pytest.raises(ValueError, match="names 1 input axes but blocks have 2"):
            BandpowerWindow(blocks=np.ones((1, 2, 3, 4)), block_index=[[0]])

    def test_window_in_coords_validation(self):
        """Coordinates must name an input axis and have shape (Npols, axis length)."""
        with pytest.raises(ValueError, match="not in in_axes"):
            BandpowerWindow(
                blocks=np.ones((1, 2, 2)), block_index=[[0]], in_coords={"k": [[1, 2]]}
            )
        with pytest.raises(ValueError, match=r"must have shape \(Npols, 2\)"):
            BandpowerWindow(
                blocks=np.ones((1, 2, 2)),
                block_index=[[0]],
                in_coords={"delay": [1, 2]},
            )

    def test_covariance_validation(self):
        """Covariance blocks must be square and scale must match block_index."""
        with pytest.raises(ValueError, match="must be square"):
            BandpowerCovariance(blocks=np.ones((1, 2, 3)), block_index=[[0]])
        with pytest.raises(ValueError, match="scale must have the shape"):
            BandpowerCovariance(
                blocks=np.ones((1, 2, 2)), block_index=[[0]], scale=[1.0, 2.0]
            )

    def test_immutable(self, exact_window):
        """Attributes cannot be reassigned; operations return new objects."""
        with pytest.raises(AttributeError):
            exact_window.blocks = np.zeros(1)


class TestAccess:
    def test_get_matches_dense(self, window_dense):
        """get() returns what indexing the dense array would."""
        window = BandpowerWindow.from_dense(window_dense)
        inds = [0, 5, 11]
        np.testing.assert_array_equal(window.get(inds, 1), window_dense[inds, ..., 1])
        np.testing.assert_array_equal(window.get(3, 0), window_dense[3, ..., 0])
        np.testing.assert_array_equal(
            window.get(slice(2, 6), 0), window_dense[2:6, ..., 0]
        )

    def test_covariance_get_applies_scale(self, cov_dense, scaled_cov):
        """Covariance get() and to_dense() multiply each entry by its scale."""
        scale = scaled_cov.scale
        np.testing.assert_allclose(scaled_cov.get(4, 1), 10.0 * cov_dense[4, ..., 1])
        np.testing.assert_allclose(
            scaled_cov.get([0, 4], 1),
            scale[[0, 4], 1, None, None] * cov_dense[[0, 4], ..., 1],
        )
        np.testing.assert_allclose(
            scaled_cov.to_dense(), cov_dense * scale[:, None, None, :]
        )


class TestStructure:
    def test_deduplicate_drops_unused_and_merges_equal(self):
        """Equal blocks merge and unreferenced blocks are dropped."""
        blocks = np.array([[[1.0]], [[2.0]], [[1.0]], [[3.0]]])
        kernel = BandpowerKernel(blocks=blocks, block_index=[[0, 2], [2, 1]])
        deduped = kernel.deduplicate()
        np.testing.assert_array_equal(deduped.blocks, [[[1.0]], [[2.0]]])
        np.testing.assert_array_equal(deduped.block_index, [[0, 0], [0, 1]])
        assert deduped == kernel

    def test_select(self, window_dense):
        """Selecting entries keeps only the blocks they use."""
        window = BandpowerWindow.from_dense(window_dense)
        bltpairs = np.array([1, 4, 7])  # blpair 1 at times 0, 1, 2
        sub = window.select(bltpairs, [1])
        assert (sub.Nbltpairs, sub.Npols) == (3, 1)
        assert sub.Nblocks == 1
        np.testing.assert_array_equal(sub.to_dense(), window_dense[bltpairs][..., [1]])

    @pytest.mark.parametrize(
        "selection", [None, slice(2, 8), np.arange(NBLTPAIRS) % 3 == 0, [0, 11, 5], 7]
    )
    def test_select_selection_types(self, window_dense, selection):
        """select() accepts None, slices, boolean masks, int lists and single ints."""
        window = BandpowerWindow.from_dense(window_dense)
        rows = np.arange(NBLTPAIRS)
        expected = window_dense[
            rows if selection is None else np.atleast_1d(rows[selection])
        ]
        np.testing.assert_array_equal(window.select(selection).to_dense(), expected)

    def test_select_subsets_coords_and_scale(self, exact_window, scaled_cov):
        """select() also cuts down per-pol coordinates and per-entry scales."""
        sub = exact_window.select(pol_inds=[1])
        for name in ("kperp", "kpara"):
            np.testing.assert_array_equal(
                sub.in_coords[name], exact_window.in_coords[name][[1]]
            )
        np.testing.assert_array_equal(
            scaled_cov.select([2, 3], [0]).scale, scaled_cov.scale[[2, 3]][:, [0]]
        )

    def test_concatenate_bltpairs(self, window_dense):
        """Splitting along bltpairs and joining back gives the original windows."""
        window = BandpowerWindow.from_dense(window_dense)
        joined = BandpowerWindow.concatenate(
            [window.select(slice(0, 5)), window.select(slice(5, None))]
        )
        assert joined == window
        assert joined.deduplicate().Nblocks == window.Nblocks

    def test_concatenate_pols(self, exact_window):
        """Joining along pols also joins the per-pol coordinates."""
        joined = BandpowerWindow.concatenate(
            [exact_window.select(pol_inds=[0]), exact_window.select(pol_inds=[1])],
            axis="pol",
        )
        assert joined == exact_window

    @pytest.mark.parametrize("axis", ["bltpair", "pol"])
    def test_concatenate_covariance_keeps_scale(self, cov_dense, scaled_cov, axis):
        """Joining covariances joins their scales along the same axis."""
        unscaled = BandpowerCovariance.from_dense(cov_dense)
        joined = BandpowerCovariance.concatenate([unscaled, scaled_cov], axis=axis)
        np.testing.assert_allclose(
            joined.to_dense(),
            np.concatenate(
                [unscaled.to_dense(), scaled_cov.to_dense()],
                axis=0 if axis == "bltpair" else -1,
            ),
        )

    @pytest.mark.parametrize(
        ("others", "axis", "error", "match"),
        [
            ([], "bltpair", ValueError, "at least one"),
            (["self"], "time", ValueError, "axis must be"),
            (["cov"], "bltpair", TypeError, "can only concatenate"),
            (["square"], "bltpair", ValueError, "block shapes"),
            (["moved_coords"], "bltpair", ValueError, "in_coords differ"),
            (["no_coords"], "bltpair", ValueError, "different in_coords"),
            (["other_axes"], "bltpair", ValueError, "in_axes"),
        ],
    )
    def test_concatenate_errors(self, exact_window, others, axis, error, match):
        """Windows that cannot be joined meaningfully raise instead."""
        options = {
            "self": exact_window,
            "cov": BandpowerCovariance(blocks=np.ones((1, 2, 2)), block_index=[[0, 0]]),
            "square": BandpowerWindow(blocks=np.ones((1, 2, 2)), block_index=[[0, 0]]),
            "moved_coords": BandpowerWindow(
                blocks=exact_window.blocks,
                block_index=exact_window.block_index,
                in_axes=exact_window.in_axes,
                in_coords={k: v + 1 for k, v in exact_window.in_coords.items()},
            ),
            "no_coords": BandpowerWindow(
                blocks=exact_window.blocks,
                block_index=exact_window.block_index,
                in_axes=exact_window.in_axes,
            ),
            "other_axes": BandpowerWindow(
                blocks=exact_window.blocks,
                block_index=exact_window.block_index,
                in_axes=("a", "b"),
            ),
        }
        kernels = [exact_window] + [options[name] for name in others] if others else []
        with pytest.raises(error, match=match):
            BandpowerWindow.concatenate(kernels, axis=axis)

    def test_concatenate_empty_covariance_list(self):
        """Joining no covariances raises."""
        with pytest.raises(ValueError, match="at least one"):
            BandpowerCovariance.concatenate([])


class TestEquality:
    def test_layout_independent(self, cov_dense):
        """Equality compares entries, not how blocks and scales are laid out."""
        cov = BandpowerCovariance.from_dense(cov_dense)
        doubled = BandpowerCovariance(
            blocks=cov.blocks / 2, block_index=cov.block_index, scale=2 * cov.scale
        )
        assert doubled == cov
        assert BandpowerCovariance.from_dense(cov_dense, deduplicate=False) == cov

    def test_unequal(self, window_dense, cov_dense, exact_window, scaled_cov):
        """Any differing entry, shape, type, scale or coordinate makes objects unequal."""
        window = BandpowerWindow.from_dense(window_dense)
        changed = window_dense.copy()
        changed[3, 0, 0, 1] += 1.0
        assert window != BandpowerWindow.from_dense(changed)
        assert window != window.select(slice(0, 5))
        assert window != BandpowerCovariance.from_dense(cov_dense)
        assert window != "not a kernel"
        assert scaled_cov != BandpowerCovariance.from_dense(cov_dense)
        assert scaled_cov != scaled_cov.select(slice(0, 5))
        moved = BandpowerWindow(
            blocks=exact_window.blocks,
            block_index=exact_window.block_index,
            in_axes=exact_window.in_axes,
            in_coords={k: v + 1 for k, v in exact_window.in_coords.items()},
        )
        assert moved != exact_window

    def test_unhashable(self, exact_window):
        """Objects holding mutable arrays are not hashable."""
        with pytest.raises(TypeError):
            hash(exact_window)


class TestBaseKernel:
    def test_operations(self, tmp_path, window_dense):
        """The base class supports the same operations without subclass attributes."""
        dense = window_dense
        kernel = BandpowerKernel.from_dense(dense)
        assert kernel.select([0, 3], [1]) == BandpowerKernel.from_dense(
            dense[[0, 3]][..., [1]]
        )
        joined = BandpowerKernel.concatenate(
            [kernel.select(slice(0, 4)), kernel.select(slice(4, None))]
        )
        assert joined == kernel
        assert kernel != BandpowerKernel.from_dense(dense[:4])
        assert kernel != "not a kernel"
        with h5py.File(tmp_path / "kernel.h5", "w") as f:
            kernel.write_hdf5(f.create_group("kernel"))
        with h5py.File(tmp_path / "kernel.h5", "r") as f:
            assert BandpowerKernel.read_hdf5(f["kernel"], slice(2, 7)) == kernel.select(
                slice(2, 7)
            )


class TestHDF5:
    def test_round_trip(self, tmp_path, exact_window, scaled_cov):
        """Writing and reading back gives equal objects, scales included."""
        with h5py.File(tmp_path / "kernels.h5", "w") as f:
            exact_window.write_hdf5(f.create_group("window"))
            scaled_cov.write_hdf5(f.create_group("cov"))
        with h5py.File(tmp_path / "kernels.h5", "r") as f:
            assert BandpowerWindow.read_hdf5(f["window"]) == exact_window
            read_cov = BandpowerCovariance.read_hdf5(f["cov"])
        assert read_cov == scaled_cov
        np.testing.assert_array_equal(read_cov.scale, scaled_cov.scale)

    @pytest.mark.parametrize(
        ("bltpairs", "pols"),
        [
            (slice(2, 9), None),
            ([1, 5, 10], [1]),
            ([10, 1, 5], 0),
            (np.arange(NBLTPAIRS) % 2 == 0, [0, 1]),
            ([], None),
        ],
    )
    def test_partial_read(self, tmp_path, window_dense, scaled_cov, bltpairs, pols):
        """Reading a selection from disk matches selecting after a full read."""
        window = BandpowerWindow.from_dense(window_dense)
        with h5py.File(tmp_path / "kernels.h5", "w") as f:
            window.write_hdf5(f.create_group("window"))
            scaled_cov.write_hdf5(f.create_group("cov"))
        with h5py.File(tmp_path / "kernels.h5", "r") as f:
            read = BandpowerWindow.read_hdf5(f["window"], bltpairs, pols)
            read_cov = BandpowerCovariance.read_hdf5(f["cov"], bltpairs, pols)
        expected = window.select(bltpairs, pols)
        assert read == expected
        assert read.Nblocks == expected.Nblocks
        assert read_cov == scaled_cov.select(bltpairs, pols)

    def test_wrong_type(self, tmp_path, exact_window):
        """Reading a group as the wrong class raises."""
        with h5py.File(tmp_path / "window.h5", "w") as f:
            exact_window.write_hdf5(f.create_group("window"))
        with h5py.File(tmp_path / "window.h5", "r") as f:
            with pytest.raises(ValueError, match="holds a 'BandpowerWindow'"):
                BandpowerCovariance.read_hdf5(f["window"])

    def test_refuses_non_empty_group(self, tmp_path, exact_window):
        """Writing into a group that already holds data raises rather than mixing."""
        with h5py.File(tmp_path / "window.h5", "w") as f:
            group = f.create_group("window")
            exact_window.write_hdf5(group)
            with pytest.raises(ValueError, match="is not empty"):
                exact_window.write_hdf5(group)
            other = f.create_group("other")
            other.create_dataset("x", data=[1])
            with pytest.raises(ValueError, match="is not empty"):
                exact_window.write_hdf5(other)
