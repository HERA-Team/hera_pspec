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
    blocks = rng.random((NBLPAIRS, NPOLS, NDLYS, NDLYS))
    return repeat_over_times(blocks @ np.swapaxes(blocks, -1, -2))


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


def dense_window_average(dense, groups, weights):
    """Reference implementation of the window averaging in grouping.average_spectra."""
    out = []
    for members in groups:
        w = np.moveaxis(weights[members], 1, -1)  # (n, Nout, Npols)
        wsum = w.sum(axis=0).clip(1e-40, np.inf)
        out.append(np.sum(dense[members] * w[:, :, None, :], axis=0) / wsum[:, None, :])
    return np.array(out)


def dense_cov_average(dense, groups, weights):
    """Reference implementation of the covariance averaging in grouping.average_spectra."""
    out = []
    for members in groups:
        w = np.moveaxis(weights[members], 1, -1)  # (n, Nout, Npols)
        wsum = w.sum(axis=0).clip(1e-40, np.inf)
        total = np.sum(dense[members] * w[:, :, None, :] * w[:, None, :, :], axis=0)
        out.append(total / wsum[:, None, :] / wsum[None, :, :])
    return np.array(out)


class TestConstruction:
    def test_from_dense_deduplicates_repeated_times(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense)
        assert window.Nbltpairs == NBLTPAIRS
        assert window.Npols == NPOLS
        assert window.Nblocks == NBLPAIRS * NPOLS
        assert window.block_shape == (NDLYS, NDLYS)
        assert window.Nout == NDLYS
        np.testing.assert_array_equal(window.to_dense(), window_dense)

    def test_from_dense_without_deduplication(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense, deduplicate=False)
        assert window.Nblocks == NBLTPAIRS * NPOLS
        assert window == BandpowerWindow.from_dense(window_dense)

    def test_from_dense_exact_window(self, exact_window):
        dense = exact_window.to_dense()
        assert dense.shape == (NBLTPAIRS, NDLYS, 3, 6, NPOLS)
        rebuilt = BandpowerWindow.from_dense(
            dense, in_axes=exact_window.in_axes, in_coords=exact_window.in_coords
        )
        assert rebuilt.Nblocks == NPOLS
        assert rebuilt == exact_window

    def test_covariance_default_scale(self, cov_dense):
        cov = BandpowerCovariance.from_dense(cov_dense)
        np.testing.assert_array_equal(cov.scale, np.ones((NBLTPAIRS, NPOLS)))
        np.testing.assert_allclose(cov.to_dense(), cov_dense)

    def test_repr(self, exact_window, cov_dense):
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
        with pytest.raises(ValueError, match=match):
            BandpowerKernel(**kwargs)

    def test_window_in_axes_must_match_block_ndim(self):
        with pytest.raises(ValueError, match="names 1 input axes but blocks have 2"):
            BandpowerWindow(blocks=np.ones((1, 2, 3, 4)), block_index=[[0]])

    def test_window_in_coords_validation(self):
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
        with pytest.raises(ValueError, match="must be square"):
            BandpowerCovariance(blocks=np.ones((1, 2, 3)), block_index=[[0]])
        with pytest.raises(ValueError, match="scale must have the shape"):
            BandpowerCovariance(
                blocks=np.ones((1, 2, 2)), block_index=[[0]], scale=[1.0, 2.0]
            )

    def test_immutable(self, exact_window):
        with pytest.raises(AttributeError):
            exact_window.blocks = np.zeros(1)


class TestAccess:
    def test_get_matches_dense(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense)
        inds = [0, 5, 11]
        np.testing.assert_array_equal(window.get(inds, 1), window_dense[inds, ..., 1])
        np.testing.assert_array_equal(window.get(3, 0), window_dense[3, ..., 0])
        np.testing.assert_array_equal(
            window.get(slice(2, 6), 0), window_dense[2:6, ..., 0]
        )

    def test_covariance_get_applies_scale(self, cov_dense):
        cov = BandpowerCovariance.from_dense(cov_dense)
        scale = np.arange(1, NBLTPAIRS * NPOLS + 1, dtype=float).reshape(
            NBLTPAIRS, NPOLS
        )
        scaled = BandpowerCovariance(
            blocks=cov.blocks, block_index=cov.block_index, scale=scale
        )
        np.testing.assert_allclose(scaled.get(4, 1), 10.0 * cov_dense[4, ..., 1])
        np.testing.assert_allclose(
            scaled.get([0, 4], 1),
            scale[[0, 4], 1, None, None] * cov_dense[[0, 4], ..., 1],
        )
        np.testing.assert_allclose(
            scaled.to_dense(), cov_dense * scale[:, None, None, :]
        )


class TestStructure:
    def test_deduplicate_drops_unused_and_merges_equal(self):
        blocks = np.array([[[1.0]], [[2.0]], [[1.0]], [[3.0]]])
        kernel = BandpowerKernel(blocks=blocks, block_index=[[0, 2], [2, 1]])
        deduped = kernel.deduplicate()
        assert deduped.Nblocks == 2
        assert deduped == kernel
        np.testing.assert_array_equal(deduped.to_dense(), kernel.to_dense())

    def test_select(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense)
        bltpairs = np.array([1, 4, 7])
        sub = window.select(bltpairs, [1])
        assert (sub.Nbltpairs, sub.Npols) == (3, 1)
        # every time of blpair 1 shares one block
        assert sub.Nblocks == 1
        np.testing.assert_array_equal(sub.to_dense(), window_dense[bltpairs][..., [1]])

    @pytest.mark.parametrize(
        "selection", [None, slice(2, 8), np.arange(NBLTPAIRS) % 3 == 0, [0, 11, 5], 7]
    )
    def test_select_selection_types(self, window_dense, selection):
        window = BandpowerWindow.from_dense(window_dense)
        rows = np.arange(NBLTPAIRS)
        expected = window_dense[
            rows if selection is None else np.atleast_1d(rows[selection])
        ]
        np.testing.assert_array_equal(window.select(selection).to_dense(), expected)

    def test_select_subsets_coords_and_scale(self, exact_window, cov_dense):
        sub = exact_window.select(pol_inds=[1])
        np.testing.assert_array_equal(
            sub.in_coords["kperp"], exact_window.in_coords["kperp"][[1]]
        )
        scale = np.arange(NBLTPAIRS * NPOLS, dtype=float).reshape(NBLTPAIRS, NPOLS)
        cov = BandpowerCovariance.from_dense(cov_dense)
        cov = BandpowerCovariance(
            blocks=cov.blocks, block_index=cov.block_index, scale=scale
        )
        np.testing.assert_array_equal(
            cov.select([2, 3], [0]).scale, scale[[2, 3]][:, [0]]
        )

    def test_concatenate_bltpairs(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense)
        joined = BandpowerWindow.concatenate(
            [window.select(slice(0, 5)), window.select(slice(5, None))]
        )
        assert joined == window
        assert joined.deduplicate().Nblocks == window.Nblocks

    def test_concatenate_pols(self, exact_window):
        joined = BandpowerWindow.concatenate(
            [exact_window.select(pol_inds=[0]), exact_window.select(pol_inds=[1])],
            axis="pol",
        )
        assert joined == exact_window

    def test_concatenate_covariance_keeps_scale(self, cov_dense):
        cov = BandpowerCovariance.from_dense(cov_dense)
        halved = cov.average([[i] for i in range(NBLTPAIRS)], weights=None)
        joined = BandpowerCovariance.concatenate([cov, halved])
        np.testing.assert_allclose(
            joined.to_dense(), np.concatenate([cov.to_dense(), halved.to_dense()])
        )

    @pytest.mark.parametrize(
        ("others", "axis", "error", "match"),
        [
            ([], "bltpair", ValueError, "at least one"),
            (["self"], "time", ValueError, "axis must be"),
            (["cov"], "bltpair", TypeError, "can only concatenate"),
            (["square"], "bltpair", ValueError, "block shapes"),
            (["moved_coords"], "bltpair", ValueError, "in_coords"),
            (["no_coords"], "bltpair", ValueError, "different in_coords"),
            (["other_axes"], "bltpair", ValueError, "in_axes"),
        ],
    )
    def test_concatenate_errors(self, exact_window, others, axis, error, match):
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


class TestEquality:
    def test_layout_independent(self, cov_dense):
        cov = BandpowerCovariance.from_dense(cov_dense)
        doubled = BandpowerCovariance(
            blocks=cov.blocks / 2, block_index=cov.block_index, scale=2 * cov.scale
        )
        assert doubled == cov
        assert BandpowerCovariance.from_dense(cov_dense, deduplicate=False) == cov

    def test_unequal(self, window_dense, cov_dense, exact_window):
        window = BandpowerWindow.from_dense(window_dense)
        changed = window_dense.copy()
        changed[3, 0, 0, 1] += 1.0
        assert window != BandpowerWindow.from_dense(changed)
        assert window != window.select(slice(0, 5))
        assert window != BandpowerCovariance.from_dense(cov_dense)
        assert window != "not a kernel"
        moved = BandpowerWindow(
            blocks=exact_window.blocks,
            block_index=exact_window.block_index,
            in_axes=exact_window.in_axes,
            in_coords={k: v + 1 for k, v in exact_window.in_coords.items()},
        )
        assert moved != exact_window

    def test_unhashable(self, exact_window):
        with pytest.raises(TypeError):
            hash(exact_window)


class TestAverage:
    groups = [
        np.arange(NBLTPAIRS)[i::NBLPAIRS] for i in range(NBLPAIRS)
    ]  # time average

    def test_window_shared_blocks_are_reused(self, window_dense, rng):
        window = BandpowerWindow.from_dense(window_dense)
        weights = rng.random((NBLTPAIRS, NPOLS, NDLYS)) + 0.1
        averaged = window.average(self.groups, weights)
        # time-averaging a time-independent window changes nothing
        assert averaged.Nblocks == window.Nblocks
        np.testing.assert_allclose(
            averaged.to_dense(),
            dense_window_average(window_dense, self.groups, weights),
        )

    def test_window_mixed_blocks(self, rng):
        dense = rng.random((NBLTPAIRS, NDLYS, NDLYS, NPOLS))
        window = BandpowerWindow.from_dense(dense)
        weights = rng.random((NBLTPAIRS, NPOLS, NDLYS))
        groups = [[0, 1, 2], [3], [4, 7, 11]]
        np.testing.assert_allclose(
            window.average(groups, weights).to_dense(),
            dense_window_average(dense, groups, weights),
        )

    def test_window_exact_default_weights(self, exact_window):
        averaged = exact_window.average([np.arange(NBLTPAIRS)])
        assert averaged.Nblocks == NPOLS
        assert averaged == exact_window.select([0])

    def test_window_zero_weights_match_reference(self, window_dense):
        window = BandpowerWindow.from_dense(window_dense)
        weights = np.ones((NBLTPAIRS, NPOLS, NDLYS))
        weights[self.groups[0], :, 2] = 0.0
        averaged = window.average(self.groups, weights)
        expected = dense_window_average(window_dense, self.groups, weights)
        np.testing.assert_allclose(averaged.to_dense(), expected)
        assert np.all(averaged.to_dense()[0, 2] == 0)

    def test_covariance_time_only_weights_rescale(self, cov_dense, rng):
        cov = BandpowerCovariance.from_dense(cov_dense)
        weights = np.repeat(rng.random((NBLTPAIRS, NPOLS, 1)) + 0.1, NDLYS, axis=-1)
        averaged = cov.average(self.groups, weights)
        assert averaged.Nblocks == cov.Nblocks
        np.testing.assert_allclose(
            averaged.to_dense(), dense_cov_average(cov_dense, self.groups, weights)
        )

    def test_covariance_default_weights(self, cov_dense):
        averaged = BandpowerCovariance.from_dense(cov_dense).average(self.groups)
        np.testing.assert_allclose(
            averaged.scale, np.full((NBLPAIRS, NPOLS), 1 / NTIMES)
        )

    def test_covariance_per_bandpower_weights(self, cov_dense, rng):
        cov = BandpowerCovariance.from_dense(cov_dense)
        weights = rng.random((NBLTPAIRS, NPOLS, NDLYS))
        groups = [[0, 3, 6], [1, 2], [5]]
        np.testing.assert_allclose(
            cov.average(groups, weights).to_dense(),
            dense_cov_average(cov_dense, groups, weights),
        )

    def test_covariance_existing_scale(self, cov_dense, rng):
        cov = BandpowerCovariance.from_dense(cov_dense)
        scale = rng.random((NBLTPAIRS, NPOLS)) + 0.5
        cov = BandpowerCovariance(
            blocks=cov.blocks, block_index=cov.block_index, scale=scale
        )
        weights = np.repeat(rng.random((NBLTPAIRS, NPOLS, 1)), NDLYS, axis=-1)
        np.testing.assert_allclose(
            cov.average(self.groups, weights).to_dense(),
            dense_cov_average(cov.to_dense(), self.groups, weights),
        )

    @pytest.mark.parametrize("cls", [BandpowerWindow, BandpowerCovariance])
    def test_errors(self, cls, cov_dense):
        kernel = cls.from_dense(cov_dense)
        with pytest.raises(ValueError, match="weights must have shape"):
            kernel.average([[0]], weights=np.ones((NBLTPAIRS, NPOLS)))
        with pytest.raises(ValueError, match="empty group"):
            kernel.average([[]])


class TestHDF5:
    def test_round_trip(self, tmp_path, exact_window, cov_dense):
        cov = BandpowerCovariance.from_dense(cov_dense).average(
            [[i, i + 1] for i in range(0, NBLTPAIRS, 2)]
        )
        with h5py.File(tmp_path / "kernels.h5", "w") as f:
            exact_window.write_hdf5(f.create_group("window"))
            cov.write_hdf5(f.create_group("cov"))
        with h5py.File(tmp_path / "kernels.h5", "r") as f:
            assert BandpowerWindow.read_hdf5(f["window"]) == exact_window
            read_cov = BandpowerCovariance.read_hdf5(f["cov"])
        assert read_cov == cov
        np.testing.assert_array_equal(read_cov.scale, cov.scale)

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
    def test_partial_read(self, tmp_path, window_dense, bltpairs, pols):
        window = BandpowerWindow.from_dense(window_dense)
        with h5py.File(tmp_path / "window.h5", "w") as f:
            window.write_hdf5(f.create_group("window"))
        with h5py.File(tmp_path / "window.h5", "r") as f:
            read = BandpowerWindow.read_hdf5(f["window"], bltpairs, pols)
        expected = window.select(bltpairs, pols)
        assert read == expected
        assert read.Nblocks == expected.Nblocks

    def test_wrong_type(self, tmp_path, exact_window):
        with h5py.File(tmp_path / "window.h5", "w") as f:
            exact_window.write_hdf5(f.create_group("window"))
        with h5py.File(tmp_path / "window.h5", "r") as f:
            with pytest.raises(ValueError, match="holds a 'BandpowerWindow'"):
                BandpowerCovariance.read_hdf5(f["window"])
