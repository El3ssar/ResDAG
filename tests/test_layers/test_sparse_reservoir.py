"""Sparse CSR ``weight_hh`` path for frozen reservoirs (Phase 14).

Verifies the opt-in ``sparse`` flag on :class:`~resdag.layers.cells.esn_cell.ESNCell`
and :class:`~resdag.layers.ESNLayer`:

* **Parity** — a sparse cell computes the recurrent term with a CSR spmv instead
  of the dense ``addmm``, but the two paths are the same math: the recurrent
  state and forecast track the dense path to floating-point tolerance.
* **Policy** — ``sparse=False`` (default) is byte-for-byte the dense path;
  ``sparse=True`` forces CSR; ``sparse="auto"`` gates on reservoir size and
  recurrent-matrix density.  ``sparse`` + ``trainable`` is rejected.
* **Live toggling** — flipping ``trainable`` on a live sparse cell densifies /
  re-sparsifies with the documented warning.
* **Serialization** — ``state_dict`` serialises the dense ``weight_hh`` (so a
  sparse checkpoint loads into a dense cell and vice versa) and pickle
  (``save_full``) round-trips the CSR buffer; ``.to()`` / ``.double()`` rebuild
  the CSR operator on the new device/dtype.
* **Compile fallback** — ``compile=True`` degrades gracefully to eager because
  ``torch.compile`` cannot trace sparse tensors.
* **Storage** — the CSR *operator* (the matmul operand, and the size a
  CSR-serialized checkpoint would be) is far smaller than an ``(n, n)`` dense
  matrix at low density.  This is a storage fact about the CSR buffer, **not** a
  live-cell-memory reduction: the dense ``weight_hh`` parameter stays resident as
  the canonical form, so a sparse cell's live footprint is ``dense(n²) +
  CSR(nnz)`` — slightly *larger* than dense-only.  The wall-clock speed claim
  (CSR spmv vs dense ``addmm``) is timing-sensitive, so it lives behind the
  ``benchmark`` marker (deselected from the default lane), not here.
"""

import io
import warnings

import pytest
import torch

from resdag import ESNModel, RidgeReadoutLayer
from resdag.core import reservoir_input
from resdag.layers import Concatenate, ESNLayer
from resdag.layers.cells import ESNCell
from resdag.layers.cells.esn_cell import (
    _SPARSE_AUTO_MAX_DENSITY,
    _SPARSE_AUTO_MIN_SIZE,
    _SPARSE_AUTO_MIN_SIZE_CPU,
    _SPARSE_AUTO_MIN_SIZE_CUDA,
)
from resdag.training import ESNTrainer


def _build_esn(sparse: bool | str, *, seed: int = 7, n: int = 1000) -> ESNModel:
    """A classic-style ESN (concat-input + ridge readout) in float64."""
    inp = reservoir_input(3)
    res = ESNLayer(
        n,
        feedback_size=3,
        topology="erdos_renyi",
        spectral_radius=0.9,
        seed=seed,
        sparse=sparse,
    )(inp)
    cat = Concatenate()(inp, res)
    out = RidgeReadoutLayer(n + 3, 3, alpha=1e-4, name="output")(cat)
    return ESNModel(inp, out).double()


def _reservoir_of(model: ESNModel) -> ESNLayer:
    """Return the (single) ESN reservoir layer of a model."""
    return next(m for m in model.modules() if isinstance(m, ESNLayer))


@pytest.fixture()
def lorenz_splits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(warmup, train_feedback, train_target) on a normalized Lorenz series."""
    from resdag import lorenz

    series = lorenz(3000, seed=0, normalize="standard").double()
    warmup = series[:, :1000]
    train = series[:, 1000:2000]
    target = series[:, 1001:2001]
    return warmup, train, target


class TestSparseConstruction:
    """Construction wiring, the ``sparse`` property, and the trainable guard."""

    def test_default_is_dense(self) -> None:
        """``sparse=False`` (default) leaves the CSR path inactive."""
        cell = ESNCell(reservoir_size=1200, feedback_size=3, topology="erdos_renyi")
        assert cell.sparse is False
        assert cell._weight_hh_csr is None
        assert isinstance(cell.weight_hh, torch.nn.Parameter)

    def test_sparse_true_activates_csr(self) -> None:
        """``sparse=True`` builds a CSR buffer from the dense weight."""
        cell = ESNCell(reservoir_size=1200, feedback_size=3, topology="erdos_renyi", sparse=True)
        assert cell.sparse is True
        assert cell._weight_hh_csr is not None
        assert cell._weight_hh_csr.is_sparse_csr
        # dense weight_hh is retained as the canonical form
        assert isinstance(cell.weight_hh, torch.nn.Parameter)
        assert torch.equal(cell._weight_hh_csr.to_dense(), cell.weight_hh.data)

    def test_sparse_true_forces_even_below_auto_gates(self) -> None:
        """Explicit ``sparse=True`` bypasses the size/density ``auto`` gates."""
        # small reservoir that ``auto`` would leave dense
        cell = ESNCell(reservoir_size=100, feedback_size=3, sparse=True)
        assert cell.sparse is True

    def test_trainable_plus_sparse_raises(self) -> None:
        """``sparse`` with ``trainable=True`` is rejected up front."""
        with pytest.raises(ValueError, match="frozen reservoir"):
            ESNCell(reservoir_size=100, feedback_size=3, sparse=True, trainable=True)
        with pytest.raises(ValueError, match="frozen reservoir"):
            ESNLayer(100, feedback_size=3, sparse="auto", trainable=True)

    def test_layer_exposes_sparse_property(self) -> None:
        """``ESNLayer.sparse`` delegates to the inner cell."""
        layer = ESNLayer(1200, feedback_size=3, topology="erdos_renyi", sparse=True)
        assert layer.sparse is True
        dense_layer = ESNLayer(1200, feedback_size=3, topology="erdos_renyi")
        assert dense_layer.sparse is False

    def test_repr_mentions_sparse(self) -> None:
        """A sparse cell advertises it in ``repr``; a dense one does not."""
        sparse = ESNLayer(1200, feedback_size=3, topology="erdos_renyi", sparse=True)
        dense = ESNLayer(1200, feedback_size=3, topology="erdos_renyi")
        assert "sparse=True" in repr(sparse)
        assert "sparse" not in repr(dense)


class TestAutoPolicy:
    """The ``sparse="auto"`` size/density gates."""

    def test_auto_small_reservoir_stays_dense(self) -> None:
        """Below the size gate, ``auto`` stays dense even when sparse."""
        n = _SPARSE_AUTO_MIN_SIZE - 500
        cell = ESNCell(reservoir_size=n, feedback_size=3, topology="erdos_renyi", sparse="auto")
        assert cell.sparse is False

    def test_auto_large_sparse_reservoir_activates(self) -> None:
        """Large + sparse ⇒ ``auto`` enables the CSR path."""
        cell = ESNCell(
            reservoir_size=2000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.05}),
            sparse="auto",
        )
        assert cell._recurrent_density() <= _SPARSE_AUTO_MAX_DENSITY
        assert cell.sparse is True

    def test_auto_dense_topology_stays_dense(self) -> None:
        """A near-dense topology above the size gate still stays dense."""
        cell = ESNCell(
            reservoir_size=1200,
            feedback_size=3,
            topology="complete",
            spectral_radius=0.9,
            sparse="auto",
        )
        assert cell._recurrent_density() > _SPARSE_AUTO_MAX_DENSITY
        assert cell.sparse is False


class TestAutoDeviceAware:
    """``sparse="auto"`` uses a device-dependent size gate.

    The dense-``addmm``/CSR-spmv crossover is far higher on a GPU (its dense BLAS
    is fast per FLOP), so ``auto`` must not select the *slower* sparse path in the
    ``1000 <= n < 5000`` GPU regime — measured on this hardware, dense wins there
    (n=1000/1%: 0.53x, n=2000/1%: 0.91x).  These tests pin the device-aware gate
    both as a pure unit (mocking the weight device, so they run on CPU-only CI)
    and, where a GPU is present, end-to-end.
    """

    def test_cpu_gate_is_the_lower_floor(self) -> None:
        """On CPU, ``_auto_min_size`` returns the (lower) CPU floor."""
        cell = ESNCell(reservoir_size=2000, feedback_size=3, topology="erdos_renyi")
        assert cell.weight_hh.device.type == "cpu"
        assert cell._auto_min_size() == _SPARSE_AUTO_MIN_SIZE_CPU
        assert _SPARSE_AUTO_MIN_SIZE_CUDA > _SPARSE_AUTO_MIN_SIZE_CPU

    def test_simulated_cuda_gate_keeps_regressing_regime_dense(self) -> None:
        """A simulated CUDA device keeps ``1000<=n<5000`` dense (no regression).

        Patches the recurrent weight's ``.device`` to report CUDA so the
        device-aware branch is exercised on a CPU-only machine.  At n=2000 (in the
        GPU-regressing band) ``auto`` must resolve to dense; the CPU gate would
        have (wrongly, for a GPU) fired sparse here.
        """
        from unittest.mock import patch

        cell = ESNCell(
            reservoir_size=2000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            sparse="auto",
        )
        # Sanity: the CPU decision *would* fire sparse at n=2000/1%.
        assert cell._recurrent_density() <= _SPARSE_AUTO_MAX_DENSITY
        assert cell.sparse is True

        # Now re-decide as if the weights lived on CUDA: n=2000 < 5000 ⇒ dense.
        with patch.object(
            type(cell.weight_hh),
            "device",
            new=property(lambda self: torch.device("cuda:0")),
        ):
            assert cell._auto_min_size() == _SPARSE_AUTO_MIN_SIZE_CUDA
            assert cell._should_use_sparse() is False

    def test_simulated_cuda_gate_activates_above_cuda_floor(self) -> None:
        """A simulated CUDA device still fires sparse at ``n >= 5000``."""
        from unittest.mock import patch

        cell = ESNCell(
            reservoir_size=5000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            sparse="auto",
        )
        with patch.object(
            type(cell.weight_hh),
            "device",
            new=property(lambda self: torch.device("cuda:0")),
        ):
            assert cell._auto_min_size() == _SPARSE_AUTO_MIN_SIZE_CUDA
            assert cell._should_use_sparse() is True

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_cuda_auto_does_not_fire_in_regressing_regime(self) -> None:
        """On a real GPU, ``auto`` leaves the ``1000<=n<5000`` band dense.

        This is the regression FINDING 2 flagged: the old CPU-only gate fired
        sparse at n=2000 on CUDA, where dense is faster.  Built directly on the
        GPU so the device-aware decision runs against real CUDA weights.
        """
        cell = ESNCell(
            reservoir_size=2000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            sparse="auto",
        ).cuda()
        assert cell.weight_hh.device.type == "cuda"
        assert cell.sparse is False  # dense wins in this regime on GPU

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_cuda_auto_activates_above_floor(self) -> None:
        """On a real GPU, ``auto`` fires sparse at ``n >= 5000`` (sparse wins)."""
        cell = ESNCell(
            reservoir_size=5000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            sparse="auto",
        ).cuda()
        assert cell.weight_hh.device.type == "cuda"
        assert cell.sparse is True


class TestParity:
    """Sparse and dense produce the same math to floating-point tolerance."""

    def test_recurrent_term_matches_dense(self) -> None:
        """The CSR spmv equals ``state @ weight_hh.T`` to float64 round-off."""
        common = dict(
            reservoir_size=1000,
            feedback_size=3,
            topology="erdos_renyi",
            spectral_radius=0.9,
            seed=42,
        )
        dense = ESNLayer(**common, sparse=False).double()
        sparse = ESNLayer(**common, sparse=True).double()
        assert torch.equal(dense.weight_hh, sparse.weight_hh)

        state = torch.randn(4, 1000, dtype=torch.float64)
        rd = dense.cell._recurrent(state)
        rs = sparse.cell._recurrent(state)
        assert torch.allclose(rd, rs, atol=1e-12, rtol=0.0)

    def test_teacher_forced_states_match(self) -> None:
        """A full teacher-forced sequence tracks the dense reservoir states."""
        common = dict(
            reservoir_size=1000,
            feedback_size=3,
            topology="erdos_renyi",
            spectral_radius=0.9,
            seed=42,
        )
        dense = ESNLayer(**common, sparse=False).double()
        sparse = ESNLayer(**common, sparse=True).double()
        fb = torch.randn(2, 300, 3, dtype=torch.float64)
        dense.reset_state()
        sparse.reset_state()
        out_d = dense(fb.clone())
        out_s = sparse(fb.clone())
        assert torch.allclose(out_d, out_s, atol=1e-10, rtol=0.0)

    def test_forecast_matches_dense(
        self, lorenz_splits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """A trained sparse ESN forecasts within float tolerance of the dense one."""
        warmup, train, target = lorenz_splits
        dense = _build_esn(False)
        sparse = _build_esn(True)
        for m in (dense, sparse):
            ESNTrainer(m).fit(
                warmup_inputs=(warmup,),
                train_inputs=(train,),
                targets={"output": target},
            )
        fd = dense.forecast(warmup, horizon=100)
        fs = sparse.forecast(warmup, horizon=100)
        # ~1e-5 is the acceptance bar; the real gap on a stable attractor is ~1e-10.
        assert torch.allclose(fd, fs, atol=1e-5, rtol=0.0)


class TestTrainableToggle:
    """Flipping ``trainable`` densifies / re-sparsifies the live cell."""

    def test_flip_to_trainable_densifies_with_warning(self) -> None:
        layer = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", sparse=True)
        assert layer.sparse is True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer.trainable = True
        assert layer.sparse is False
        assert layer.cell._weight_hh_csr is None
        assert any(
            issubclass(w.category, RuntimeWarning) and "densif" in str(w.message) for w in caught
        )
        # dense forward still works
        _ = layer(torch.randn(1, 10, 3))

    def test_flip_back_to_frozen_re_sparsifies(self) -> None:
        layer = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", sparse=True)
        layer.trainable = True
        layer.trainable = False
        assert layer.sparse is True
        assert layer.cell._weight_hh_csr is not None
        assert not layer.weight_hh.requires_grad


class TestSerialization:
    """state_dict / pickle round-trips and device/dtype moves."""

    def test_state_dict_serializes_dense_weight(self) -> None:
        """The CSR buffer is non-persistent; ``weight_hh`` serialises dense."""
        layer = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", sparse=True)
        sd = layer.state_dict()
        keys = [k for k in sd if k.endswith("weight_hh")]
        assert keys, "weight_hh must appear in state_dict"
        for k in keys:
            assert sd[k].layout == torch.strided  # dense, not CSR
        # the non-persistent CSR buffer is absent
        assert not any(k.endswith("_weight_hh_csr") for k in sd)

    def test_state_dict_round_trip_sparse_to_sparse(self) -> None:
        a = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=1, sparse=True)
        b = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=2, sparse=True)
        b.load_state_dict(a.state_dict())
        assert b.sparse is True
        assert torch.equal(a.weight_hh, b.weight_hh)
        assert torch.equal(a.cell._weight_hh_csr.to_dense(), b.cell._weight_hh_csr.to_dense())

    def test_state_dict_cross_mode_sparse_to_dense(self) -> None:
        """A sparse checkpoint (dense-serialized) loads into a dense cell."""
        sparse = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=1, sparse=True)
        dense = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=2, sparse=False)
        dense.load_state_dict(sparse.state_dict())
        assert dense.sparse is False
        assert torch.equal(dense.weight_hh, sparse.weight_hh)

    def test_state_dict_cross_mode_dense_to_sparse(self) -> None:
        """A dense checkpoint loads into a sparse cell and rebuilds the CSR view."""
        dense = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=1)
        sparse = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=2, sparse=True)
        sparse.load_state_dict(dense.state_dict())
        assert sparse.sparse is True
        assert torch.equal(sparse.weight_hh, dense.weight_hh)
        assert torch.equal(sparse.cell._weight_hh_csr.to_dense(), dense.weight_hh.data)

    def test_pickle_round_trip_preserves_csr(self) -> None:
        """Whole-object pickle (``save_full`` path) preserves the CSR buffer."""
        layer = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", seed=1, sparse=True)
        buf = io.BytesIO()
        torch.save(layer, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)
        assert restored.sparse is True
        assert restored.cell._weight_hh_csr.is_sparse_csr
        assert torch.equal(restored.weight_hh, layer.weight_hh)
        # and it still runs
        fb = torch.randn(1, 10, 3)
        assert torch.allclose(restored(fb), layer(fb))

    def test_double_rebuilds_csr_dtype(self) -> None:
        """``.double()`` rebuilds the CSR operator at the new dtype."""
        layer = ESNLayer(1000, feedback_size=3, topology="erdos_renyi", sparse=True)
        assert layer.cell._weight_hh_csr.dtype == torch.float32
        layer = layer.double()
        assert layer.sparse is True
        assert layer.cell._weight_hh_csr.dtype == torch.float64
        assert torch.equal(layer.cell._weight_hh_csr.to_dense(), layer.weight_hh.data)


class TestModelIntegration:
    """The factory + model paths accept ``sparse`` and forecast correctly."""

    def test_classic_esn_forwards_sparse(
        self, lorenz_splits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        from resdag.models import classic_esn

        warmup, train, target = lorenz_splits
        model = classic_esn(
            1000,
            feedback_size=3,
            output_size=3,
            topology="erdos_renyi",
            spectral_radius=0.9,
            seed=7,
            sparse=True,
        ).double()
        assert _reservoir_of(model).sparse is True
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )
        out = model.forecast(warmup, horizon=50)
        assert out.shape == (1, 50, 3)
        assert torch.isfinite(out).all()

    def test_forecast_compile_falls_back_gracefully(
        self, lorenz_splits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """``compile=True`` on a sparse model equals the eager forecast.

        ``torch.compile`` cannot trace sparse tensors, so the forecast engine
        degrades to the eager flat step and produces an identical result.
        """
        warmup, train, target = lorenz_splits
        model = _build_esn(True)
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compiled = model.forecast(warmup, horizon=40, compile=True)
        eager = model.forecast(warmup, horizon=40, compile=False)
        assert torch.allclose(compiled, eager, atol=1e-10, rtol=0.0)


def _storage_bytes(t: torch.Tensor) -> int:
    """Resident byte count of a dense or CSR tensor (values + index arrays)."""
    if t.is_sparse_csr:
        return (
            t.values().numel() * t.values().element_size()
            + t.crow_indices().numel() * t.crow_indices().element_size()
            + t.col_indices().numel() * t.col_indices().element_size()
        )
    return t.numel() * t.element_size()


class TestStorage:
    """Deterministic storage facts about the CSR operator (no wall-clock).

    These assert *what is actually measured* — the byte size of the CSR buffer
    vs a dense ``(n, n)`` matrix — and the honest live-memory relationship (the
    dense ``weight_hh`` stays resident, so the CSR is *additional* storage).  The
    wall-clock speed claim lives in :class:`TestSpmvSpeed` behind the
    ``benchmark`` marker, because timing is machine-load-sensitive.
    """

    def test_csr_operator_smaller_than_dense_at_low_density(self) -> None:
        """At n=5000, density≈1%, the CSR *operator* is far smaller than dense.

        This is a fact about the CSR buffer (the matmul operand, and the size a
        CSR-serialized checkpoint would be), **not** about live cell memory — the
        dense ``weight_hh`` remains resident alongside it (see
        :meth:`test_sparse_cell_live_memory_exceeds_dense`).
        """
        layer = ESNLayer(
            5000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            spectral_radius=0.9,
            seed=0,
            sparse=True,
        )
        dense_bytes = _storage_bytes(layer.weight_hh.data)
        csr_bytes = _storage_bytes(layer.cell._weight_hh_csr)
        # CSR must be at least 10x smaller than the dense (n,n) matrix at 1%.
        assert csr_bytes < dense_bytes / 10

    def test_sparse_cell_live_memory_exceeds_dense(self) -> None:
        """Honest memory accounting: a sparse cell is *larger* live than dense.

        ``sparse=True`` does **not** lift the memory ceiling: the canonical dense
        ``weight_hh`` parameter is always materialised, and the CSR buffer is an
        *additional* non-persistent operator built from it.  So the sparse cell's
        resident recurrent storage is ``dense(n²) + CSR(nnz)`` — strictly greater
        than the dense-only cell's ``dense(n²)``.  This test pins that reality so
        the feature's memory claim can never silently regress into overstatement.
        """
        common = dict(
            reservoir_size=2000,
            feedback_size=3,
            topology=("erdos_renyi", {"p": 0.01}),
            spectral_radius=0.9,
            seed=0,
        )
        dense = ESNCell(**common, sparse=False)
        sparse = ESNCell(**common, sparse=True)

        def resident(cell: ESNCell) -> int:
            total = sum(_storage_bytes(p.data) for p in cell.parameters())
            for buf in cell.buffers():
                if buf is not None:
                    total += _storage_bytes(buf)
            return total

        dense_bytes = resident(dense)
        sparse_bytes = resident(sparse)
        # The sparse cell carries the same dense weight_hh PLUS the CSR buffer,
        # so it is strictly larger, by exactly the CSR operator's size.
        assert sparse_bytes > dense_bytes
        assert sparse._weight_hh_csr is not None
        csr_bytes = _storage_bytes(sparse._weight_hh_csr)
        assert sparse_bytes - dense_bytes == csr_bytes


@pytest.mark.benchmark
class TestSpmvSpeed:
    """The CSR spmv beats the dense ``addmm`` at RC densities (timing gate).

    Behind the ``benchmark`` marker — deselected from the default lane
    (``pyproject.toml`` ``addopts: -m 'not benchmark'``) — because it is a
    wall-clock assertion and the normal lane must stay deterministic under
    arbitrary machine load.  Run with ``pytest -m benchmark``.
    """

    @staticmethod
    def _best_ratio(
        slow: "callable", fast: "callable", *, warmup: int = 3, reps: int = 9
    ) -> tuple[float, float, float]:
        """Best ``slow / fast`` ratio from *interleaved* A/B timing.

        Each rep times ``slow`` then ``fast`` back-to-back so both see the same
        instantaneous load; the maximum ratio over reps (the cleanest rep) is
        returned.  This is the same robust pattern the repo's other timing gate
        uses (``tests/test_benchmarks/test_perf_regression.py``): interleaving +
        best-ratio isolates the intrinsic speedup from transient load asymmetry.
        """
        import time

        for _ in range(warmup):
            slow()
            fast()
        best_ratio = 0.0
        best_slow = 0.0
        best_fast = 0.0
        for _ in range(reps):
            s0 = time.perf_counter()
            slow()
            t_slow = time.perf_counter() - s0
            f0 = time.perf_counter()
            fast()
            t_fast = time.perf_counter() - f0
            ratio = t_slow / t_fast if t_fast > 0 else 0.0
            if ratio > best_ratio:
                best_ratio, best_slow, best_fast = ratio, t_slow, t_fast
        return best_ratio, best_slow, best_fast

    @pytest.mark.parametrize("n,density", [(2000, 0.01), (2000, 0.05), (5000, 0.01)])
    def test_sparse_spmv_faster_than_dense_addmm(self, n: int, density: float) -> None:
        """CSR recurrent term beats the dense ``addmm`` at RC densities on CPU.

        Timed on the isolated recurrent matmul (the hot inner-loop kernel) with
        interleaved best-of-N A/B timing and a conservative ``1.2x`` floor.  The
        floor is deliberately low: the *measured* CPU speedups at these
        ``(n, density)`` points are large (n=2000/1% ~14x, n=5000/1% ~24x), so a
        1.2x gate still fails loudly if sparse regresses toward par, while the
        interleaving keeps it robust to CI jitter.  The gate runs CPU-only —
        the dense/sparse crossover on CUDA is at much larger ``n`` (see the
        device-aware ``auto`` policy in ``esn_cell.py``).
        """
        g = torch.Generator().manual_seed(0)
        w = torch.zeros(n, n, dtype=torch.float64)
        mask = torch.rand(n, n, generator=g) < density
        w[mask] = torch.randn(int(mask.sum()), dtype=torch.float64)
        csr = w.to_sparse_csr()
        wt = w.t().contiguous()
        state = torch.randn(1, n, dtype=torch.float64)
        proj = torch.randn(1, n, dtype=torch.float64)

        def dense() -> torch.Tensor:
            return torch.addmm(proj, state, wt)

        def sparse() -> torch.Tensor:
            return proj + torch.sparse.mm(csr, state.t().contiguous()).t()

        ratio, t_dense, t_sparse = self._best_ratio(dense, sparse)
        assert ratio > 1.2, (
            f"sparse spmv not faster: best dense/sparse ratio {ratio:.2f}x "
            f"(dense {t_dense * 1e3:.3f}ms vs sparse {t_sparse * 1e3:.3f}ms; "
            f"n={n}, density={density})"
        )
