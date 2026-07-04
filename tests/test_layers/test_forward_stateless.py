"""Tests for the pure ``forward_stateless`` reservoir loop (issue #253).

``BaseReservoirLayer.forward_stateless(inputs, state) -> (outputs, new_state)``
is the functional core that the stateful :meth:`forward` wraps.  These tests
pin the contract that makes it compile-scan / vmap / export friendly:

- it is **pure**: it never reads or writes ``self.state`` and never mutates the
  input ``state`` in place (calling it twice yields identical results);
- it is **value-preserving**: the stateful ``forward`` it now drives produces
  output identical to the previous preallocate-and-in-place-write loop;
- it **threads state**: feeding the returned ``new_state`` back in continues the
  trajectory seamlessly;
- the cross-call ``grad_fn`` detach lives in the wrapper, not the pure loop.

``NGReservoir`` overrides it with the vectorized ``NGCell.forward_sequence`` and
is held to the same contract.
"""

import pytest
import torch

from resdag.layers import ESNLayer, NGReservoir


def _legacy_forward(
    layer: ESNLayer,
    feedback: torch.Tensor,
    *driving_inputs: torch.Tensor,
) -> torch.Tensor:
    """Reference: the pre-refactor stateful loop (preallocate + in-place writes).

    Faithfully replicates the implementation ``forward`` had before
    ``forward_stateless`` was introduced — a ``torch.empty`` output buffer
    written slice-by-slice — starting from a fresh zero state, so the new
    ``forward`` / ``forward_stateless`` can be checked bit-for-bit against it.
    """
    cell = layer.cell
    batch_size, seq_len, _ = feedback.shape
    state = cell.init_state(batch_size, feedback.device, feedback.dtype)
    outputs = torch.empty(
        batch_size,
        seq_len,
        cell.output_size,
        device=feedback.device,
        dtype=feedback.dtype,
    )
    projected = cell.project_inputs([feedback, *driving_inputs])
    if projected is not None:
        for t in range(seq_len):
            output, state = cell.step(projected[:, t, :], state)
            outputs[:, t, :] = output
    else:
        for t in range(seq_len):
            inputs_t = [feedback[:, t, :], *(di[:, t, :] for di in driving_inputs)]
            output, state = cell(inputs_t, state)
            outputs[:, t, :] = output
    return outputs


class TestForwardMatchesLegacy:
    """The refactored ``forward`` is value-identical to the old loop."""

    @pytest.mark.parametrize("leak_rate", [1.0, 0.3])
    @pytest.mark.parametrize("bias", [True, False])
    def test_feedback_only(self, device: torch.device, leak_rate: float, bias: bool) -> None:
        layer = ESNLayer(
            reservoir_size=64,
            feedback_size=5,
            spectral_radius=0.9,
            leak_rate=leak_rate,
            bias=bias,
            seed=0,
        ).to(device)
        layer.eval()
        feedback = torch.randn(4, 30, 5, device=device)

        layer.reset_state()
        new = layer(feedback)
        legacy = _legacy_forward(layer, feedback)

        assert new.shape == (4, 30, 64)
        assert torch.equal(new, legacy)

    def test_with_driving_input(self, device: torch.device) -> None:
        layer = ESNLayer(
            reservoir_size=48,
            feedback_size=3,
            input_size=7,
            spectral_radius=0.9,
            leak_rate=0.5,
            seed=1,
        ).to(device)
        layer.eval()
        feedback = torch.randn(2, 25, 3, device=device)
        driver = torch.randn(2, 25, 7, device=device)

        layer.reset_state()
        new = layer(feedback, driver)
        legacy = _legacy_forward(layer, feedback, driver)

        assert torch.equal(new, legacy)

    def test_forward_equals_forward_stateless(self, device: torch.device) -> None:
        """The wrapper output equals the pure path started from a zero state."""
        layer = ESNLayer(40, feedback_size=3, spectral_radius=0.9, seed=3).to(device)
        layer.eval()
        feedback = torch.randn(2, 25, 3, device=device)

        layer.reset_state()
        wrapper_out = layer(feedback)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)
        pure_out, _ = layer.forward_stateless([feedback], state0)

        assert torch.equal(wrapper_out, pure_out)


class TestForwardStatelessPurity:
    """It touches neither ``self.state`` nor the input ``state``."""

    def test_does_not_touch_self_state(self) -> None:
        layer = ESNLayer(48, feedback_size=4, spectral_radius=0.9, seed=0)
        layer.eval()
        feedback = torch.randn(3, 20, 4)
        state0 = layer.cell.init_state(3, feedback.device, feedback.dtype)

        assert layer.state is None
        layer.forward_stateless([feedback], state0)
        assert layer.state is None  # never lazily initialised by the pure path

        # ... even when self.state already holds a warmed-up trajectory.
        layer(feedback)
        assert layer.state is not None
        saved = layer.state.clone()
        layer.forward_stateless([feedback], state0)
        assert torch.equal(layer.state, saved)

    def test_repeatable_and_input_state_unmutated(self) -> None:
        layer = ESNLayer(48, feedback_size=4, spectral_radius=0.9, seed=0)
        layer.eval()
        feedback = torch.randn(3, 20, 4)
        state0 = layer.cell.init_state(3, feedback.device, feedback.dtype)
        state0_clone = state0.clone()

        out1, ns1 = layer.forward_stateless([feedback], state0)
        out2, ns2 = layer.forward_stateless([feedback], state0)

        assert torch.equal(out1, out2)
        assert torch.equal(ns1, ns2)
        # The input state must not have been written in place.
        assert torch.equal(state0, state0_clone)

    def test_returns_final_state_shape(self) -> None:
        layer = ESNLayer(32, feedback_size=3, seed=5)
        layer.eval()
        feedback = torch.randn(2, 10, 3)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)

        outputs, new_state = layer.forward_stateless([feedback], state0)

        assert outputs.shape == (2, 10, 32)
        assert new_state.shape == (2, 32)
        # Final output row equals the returned final state (ESN: output == state).
        assert torch.equal(outputs[:, -1, :], new_state)


class TestForwardStatelessThreadsState:
    """Feeding ``new_state`` back in continues the trajectory."""

    def test_split_sequence_matches_full(self, device: torch.device) -> None:
        layer = ESNLayer(48, feedback_size=4, spectral_radius=0.9, leak_rate=0.7, seed=1).to(device)
        layer.eval()
        feedback = torch.randn(2, 40, 4, device=device)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)

        full_out, full_state = layer.forward_stateless([feedback], state0)

        first, second = feedback[:, :15], feedback[:, 15:]
        out_a, state_a = layer.forward_stateless([first], state0)
        out_b, state_b = layer.forward_stateless([second], state_a)

        assert torch.allclose(torch.cat([out_a, out_b], dim=1), full_out, atol=1e-6)
        assert torch.allclose(state_b, full_state, atol=1e-6)


class TestForwardStatelessDetachContract:
    """The cross-call detach lives in the wrapper, not the pure loop."""

    def test_pure_path_keeps_grad_fn(self) -> None:
        layer = ESNLayer(32, feedback_size=3, trainable=True, seed=4)
        feedback = torch.randn(2, 10, 3)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)

        _, new_state = layer.forward_stateless([feedback], state0)

        # forward_stateless does NOT detach — gradients flow through new_state.
        assert new_state.grad_fn is not None

    def test_wrapper_detaches_stored_state(self) -> None:
        layer = ESNLayer(32, feedback_size=3, trainable=True, seed=4)
        feedback = torch.randn(2, 10, 3)

        layer.reset_state()
        out = layer(feedback)

        assert out.requires_grad
        assert layer.state is not None
        assert layer.state.grad_fn is None  # detached by the wrapper

    def test_differentiable_no_inplace_break(self) -> None:
        """Gradients flow to the inputs; the stacked output backprops cleanly."""
        layer = ESNLayer(32, feedback_size=3, spectral_radius=0.9, trainable=True, seed=2)
        feedback = torch.randn(2, 12, 3, requires_grad=True)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)

        out, _ = layer.forward_stateless([feedback], state0)
        out.sum().backward()

        assert feedback.grad is not None
        assert torch.isfinite(feedback.grad).all()


class TestForwardStatelessEmptySequence:
    """A zero-length sequence is handled without ``torch.stack`` choking."""

    def test_empty_sequence(self) -> None:
        layer = ESNLayer(16, feedback_size=3, seed=6)
        feedback = torch.randn(2, 0, 3)
        state0 = layer.cell.init_state(2, feedback.device, feedback.dtype)

        outputs, new_state = layer.forward_stateless([feedback], state0)

        assert outputs.shape == (2, 0, 16)
        # The state is returned unchanged for an empty input.
        assert torch.equal(new_state, state0)


class TestNGReservoirForwardStateless:
    """NG-RC overrides ``forward_stateless`` with the vectorized feature map."""

    def test_matches_forward(self, device: torch.device) -> None:
        layer = NGReservoir(input_dim=3, k=3, s=2, p=2).to(device)
        x = torch.randn(2, 50, 3, device=device)

        layer.reset_state()
        fwd = layer(x)
        state0 = layer.cell.init_state(2, x.device, x.dtype)
        pure, _ = layer.forward_stateless([x], state0)

        assert pure.shape == (2, 50, layer.cell.feature_dim)
        assert torch.equal(fwd, pure)

    def test_is_pure(self) -> None:
        layer = NGReservoir(input_dim=3, k=2, p=2)
        x = torch.randn(2, 30, 3)
        state0 = layer.cell.init_state(2, x.device, x.dtype)
        state0_clone = state0.clone()

        assert layer.state is None
        out1, ns1 = layer.forward_stateless([x], state0)
        out2, ns2 = layer.forward_stateless([x], state0)

        assert layer.state is None
        assert torch.equal(out1, out2)
        assert torch.equal(ns1, ns2)
        assert torch.equal(state0, state0_clone)  # input delay buffer not mutated

    def test_threads_state_across_split(self) -> None:
        """The carried delay buffer makes a split sequence reconstruct the whole."""
        layer = NGReservoir(input_dim=2, k=3, s=2, p=2)
        x = torch.randn(1, 60, 2)
        state0 = layer.cell.init_state(1, x.device, x.dtype)

        full, full_state = layer.forward_stateless([x], state0)

        first, second = x[:, :25], x[:, 25:]
        out_a, state_a = layer.forward_stateless([first], state0)
        out_b, state_b = layer.forward_stateless([second], state_a)

        assert torch.allclose(torch.cat([out_a, out_b], dim=1), full, atol=1e-6)
        assert torch.allclose(state_b, full_state, atol=1e-6)


# The compiled teacher-forced scan targets the CUDA reduce-overhead / cudagraph
# path (its whole benefit is shedding per-step *GPU* launch overhead), so its
# equivalence and cache-population assertions run only on a GPU.  ``atol``/``rtol``
# are tight rather than bitwise: chaotic reservoir dynamics under torch.compile
# need not be bit-exact with eager (empirically the recurrent-only ESN step is,
# but the contract is allclose).
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestForwardStatelessCompiledScan:
    """Opt-in ``compile=True`` engine matches eager and falls back cleanly."""

    @_requires_cuda
    @pytest.mark.parametrize("seq_len", [200, 130, 63])
    def test_compiled_matches_eager_feedback_only(self, seq_len: int) -> None:
        """Chunked compiled scan (incl. < chunk remainder) equals the eager loop."""
        layer = ESNLayer(128, feedback_size=3, spectral_radius=0.9, seed=0).to("cuda")
        layer.eval()
        feedback = torch.randn(1, seq_len, 3, device="cuda")

        with torch.no_grad():
            layer.reset_state()
            eager = layer(feedback)
            layer.reset_state()
            compiled = layer(feedback, compile=True)

        assert compiled.shape == eager.shape
        assert torch.allclose(eager, compiled, atol=1e-5, rtol=1e-5)

    @_requires_cuda
    def test_compiled_matches_eager_with_driver(self) -> None:
        layer = ESNLayer(96, feedback_size=3, input_size=4, spectral_radius=0.9, seed=2).to("cuda")
        layer.eval()
        feedback = torch.randn(1, 200, 3, device="cuda")
        driver = torch.randn(1, 200, 4, device="cuda")

        with torch.no_grad():
            layer.reset_state()
            eager = layer(feedback, driver)
            layer.reset_state()
            compiled = layer(feedback, driver, compile=True)

        assert torch.allclose(eager, compiled, atol=1e-5, rtol=1e-5)

    @_requires_cuda
    def test_compiled_final_state_matches_eager(self) -> None:
        """The threaded final state (not just per-step outputs) matches eager."""
        layer = ESNLayer(64, feedback_size=3, spectral_radius=0.9, leak_rate=0.6, seed=5).to("cuda")
        layer.eval()
        feedback = torch.randn(1, 150, 3, device="cuda")
        state0 = layer.cell.init_state(1, feedback.device, feedback.dtype)

        with torch.no_grad():
            _, eager_state = layer.forward_stateless([feedback], state0)
            _, compiled_state = layer.forward_stateless([feedback], state0, compile=True)

        assert torch.allclose(eager_state, compiled_state, atol=1e-5, rtol=1e-5)

    @_requires_cuda
    def test_compiled_scan_populates_and_reuses_cache(self) -> None:
        """First compiled call fills the per-config cache; the second reuses it."""
        layer = ESNLayer(64, feedback_size=3, spectral_radius=0.9, seed=1).to("cuda")
        layer.eval()
        feedback = torch.randn(1, 200, 3, device="cuda")

        assert len(layer._compiled_scan_cache) == 0
        with torch.no_grad():
            layer.reset_state()
            layer(feedback, compile=True)
            assert len(layer._compiled_scan_cache) == 1
            cached_fn = next(iter(layer._compiled_scan_cache.values()))

            layer.reset_state()
            layer(feedback, compile=True)
        # Same key, same compiled closure — not recompiled.
        assert len(layer._compiled_scan_cache) == 1
        assert next(iter(layer._compiled_scan_cache.values())) is cached_fn

    def test_compile_never_runs_when_grad_required(self) -> None:
        """A grad-tracked (trainable) forward stays eager and remains differentiable."""
        layer = ESNLayer(32, feedback_size=3, spectral_radius=0.9, trainable=True, seed=4)
        if torch.cuda.is_available():
            layer = layer.to("cuda")
        dev = next(layer.parameters()).device
        feedback = torch.randn(2, 40, 3, device=dev, requires_grad=True)

        layer.reset_state()
        out = layer(feedback, compile=True)  # grad enabled + trainable -> eager
        out.sum().backward()

        assert len(layer._compiled_scan_cache) == 0  # compiled path never taken
        assert feedback.grad is not None
        assert torch.isfinite(feedback.grad).all()

    def test_cpu_compile_falls_back_byte_identical(self) -> None:
        """On CPU the compiled scan is skipped silently and equals the eager path."""
        layer = ESNLayer(48, feedback_size=3, spectral_radius=0.9, seed=7)  # CPU
        layer.eval()
        feedback = torch.randn(1, 130, 3)

        with torch.no_grad():
            layer.reset_state()
            eager = layer(feedback)
            layer.reset_state()
            compiled = layer(feedback, compile=True)

        assert torch.equal(eager, compiled)  # byte-identical, no compile attempted
        assert len(layer._compiled_scan_cache) == 0

    def test_compile_false_default_is_unchanged(self, device: torch.device) -> None:
        """``compile=False`` (the default) is byte-identical to omitting it."""
        layer = ESNLayer(64, feedback_size=4, spectral_radius=0.9, seed=3).to(device)
        layer.eval()
        feedback = torch.randn(2, 60, 4, device=device)

        layer.reset_state()
        default = layer(feedback)
        layer.reset_state()
        explicit = layer(feedback, compile=False)

        assert torch.equal(default, explicit)

    @_requires_cuda
    def test_compiled_scan_dropped_on_deepcopy_and_pickle(self) -> None:
        """The compiled closure is dropped on copy/pickle and rebuilt lazily."""
        import copy
        import pickle

        layer = ESNLayer(64, feedback_size=3, spectral_radius=0.9, seed=8).to("cuda")
        layer.eval()
        feedback = torch.randn(1, 200, 3, device="cuda")
        with torch.no_grad():
            layer.reset_state()
            layer(feedback, compile=True)
        assert len(layer._compiled_scan_cache) == 1

        clone = copy.deepcopy(layer)
        assert len(clone._compiled_scan_cache) == 0

        restored = pickle.loads(pickle.dumps(layer))
        assert len(restored._compiled_scan_cache) == 0

        # The rebuilt-lazily contract: the restored layer still forwards, and a
        # fresh compiled call repopulates its own (independent) cache.
        with torch.no_grad():
            restored.reset_state()
            out = restored(feedback, compile=True)
        assert out.shape == (1, 200, 64)
        assert len(restored._compiled_scan_cache) == 1
