"""08 — Save / load: persistence patterns for ESN models.

Two persistence styles:

* ``ESNModel.save()`` stores the state dict (weights), optional reservoir
  states, and arbitrary metadata. The architecture itself is NOT stored:
  re-create the model (same factory call / same DAG), then load into it.
* ``ESNModel.save_full()`` pickles the *whole* model — architecture,
  weights, and states — so ``load_full()`` rebuilds nothing. This uses the
  pickling support added in ``pytorch-symbolic`` 1.2.

What it shows
-------------
1. Basic save/load roundtrip — identical outputs
2. Checkpoints with metadata and reservoir states
3. ESNModel.load_from_file class method
4. Forecast continuity: saved states resume exactly where you stopped
5. Cross-device: save on CPU, load on GPU (skipped if no CUDA)
6. Whole-model save_full/load_full — no rebuild needed

Expected runtime: ~5 s on CPU.
"""

import tempfile
from pathlib import Path

import torch

import resdag as rd
from resdag.core import ESNModel
from resdag.training import ESNTrainer


def build_model(size: int = 200) -> ESNModel:
    """The architecture must be re-created before loading weights into it."""
    torch.manual_seed(0)  # seed only affects the initial (pre-load) weights
    return rd.classic_esn(reservoir_size=size, feedback_size=3, output_size=3)


def main() -> None:
    from _example_options import parse_options

    options = parse_options(__doc__)
    torch.manual_seed(42)
    size = options.choose(24, 200)
    washout, train_end = options.choose((12, 92), (200, 800))
    horizon = options.choose(24, 100)
    tmpdir = Path(tempfile.mkdtemp(prefix="resdag_example_"))
    print(f"Writing artifacts to {tmpdir}\n")

    # A small trained model to persist
    # A bounded smooth signal keeps the persistence demonstration separate
    # from an unstable free-running fit to a random walk.
    t = 0.05 * torch.arange(options.choose(141, 1001), dtype=torch.float32)
    series = torch.stack((t.sin(), t.cos(), (0.5 * t).sin()), dim=-1).unsqueeze(0)
    model = build_model(size)
    ESNTrainer(model).fit(
        warmup_inputs=(series[:, :washout],),
        train_inputs=(series[:, washout:train_end],),
        targets={"output": series[:, washout + 1 : train_end + 1]},
    )

    # ------------------------------------------------------------------
    # 1. Basic save / load roundtrip
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Basic save / load")
    print("=" * 70)

    path = tmpdir / "model.pt"
    model.save(path)
    print(f"saved weights to {path.name} ({path.stat().st_size / 1024:.0f} KiB)")

    restored = build_model(size)  # fresh instance, different random weights
    restored.load(path)  # now identical to `model`

    x = torch.randn(2, 50, 3)  # (batch, time, features)
    model.reset_reservoirs()
    restored.reset_reservoirs()
    same = torch.allclose(model(x), restored(x))
    assert same
    print(f"restored model reproduces original outputs: {same}")

    # ------------------------------------------------------------------
    # 2. Checkpoints: metadata + reservoir states
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Checkpoints with metadata and reservoir states")
    print("=" * 70)

    model.reset_reservoirs()
    model.warmup(series[:, :washout])  # put something in the reservoir state

    ckpt = tmpdir / "checkpoint.pt"
    model.save(ckpt, include_states=True, epoch=10, val_mse=0.0123, note="demo run")
    print("saved with include_states=True and metadata kwargs")

    # Metadata is plain torch.save payload — inspect without a model:
    payload = torch.load(ckpt, weights_only=True)
    print(f"metadata: {payload['metadata']}")
    print(f"stored reservoir states: {list(payload['reservoir_states'].keys())}")

    # ------------------------------------------------------------------
    # 3. load_from_file class method
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. ESNModel.load_from_file")
    print("=" * 70)

    restored = ESNModel.load_from_file(ckpt, model=build_model(size), load_states=True)
    state = next(iter(restored.get_reservoir_states().values()))
    print(
        f"weights + states loaded; state shape {tuple(state.shape)}, "
        f"norm {state.norm().item():.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Forecast continuity through a save/load cycle
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Forecast continuity")
    print("=" * 70)
    print("Saving states mid-sequence lets a later process continue the")
    print("forecast exactly where this one stopped.")

    f_warmup = series[:, train_end - washout : train_end]

    # Reference: use the same prefix/final-frame call shapes as the resumed
    # run, so this checks persistence rather than different floating-point
    # projection schedules.
    model.reset_reservoirs()
    model.warmup(f_warmup[:, :-1])
    reference = model.forecast(f_warmup[:, -1:], horizon=horizon, reset=False)

    # Same forecast, split across a save/load boundary. Warm up on all but
    # the last sample, save, and hand the remaining sample to the resumed
    # forecast — it completes the warmup and continues autoregressively.
    model.reset_reservoirs()
    model.warmup(f_warmup[:, :-1])
    model.save(tmpdir / "warmed_up.pt", include_states=True)

    resumed = ESNModel.load_from_file(
        tmpdir / "warmed_up.pt", model=build_model(size), load_states=True
    )
    continued = resumed.forecast(f_warmup[:, -1:], horizon=horizon, reset=False)

    torch.testing.assert_close(reference, continued, atol=0, rtol=0)
    print("Full forecast continuity verified with identical call shapes.")
    print("Changing chunk shapes can change floating-point rounding; unstable")
    print("autoregressive dynamics may amplify that difference. This check isolates")
    print("the save/load boundary and does not establish forecast accuracy.")

    # ------------------------------------------------------------------
    # 5. Cross-device
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Cross-device save/load")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_model = build_model(size).to("cuda")
        gpu_model.load(path)  # CPU checkpoint loads into the CUDA model
        y = gpu_model(x.to("cuda"))
        print(f"CPU checkpoint loaded on GPU; output device: {y.device}")
    else:
        print("CUDA not available — skipped (the pattern is model.to('cuda')")
        print("then model.load(path); torch maps tensors automatically).")

    # ------------------------------------------------------------------
    # 6. Whole-model save_full / load_full (no rebuild needed)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. Whole-model save_full / load_full")
    print("=" * 70)
    print("save_full pickles the architecture too, so load_full needs no")
    print("build function — handy for quick checkpoints and experiment logs.")

    full_path = tmpdir / "model_full.pt"
    model.save_full(full_path, epoch=10, val_mse=0.0123)
    print(f"saved full model to {full_path.name} ({full_path.stat().st_size / 1024:.0f} KiB)")

    # No build_model(size) call — the architecture is reconstructed from the file.
    reloaded, meta = ESNModel.load_full(full_path, return_metadata=True, trusted=True)
    print(f"reconstructed without rebuilding; metadata: {meta}")

    model.reset_reservoirs()
    reloaded.reset_reservoirs()
    same_full = torch.allclose(model(x), reloaded(x))
    assert same_full
    print(f"reloaded full model reproduces original outputs: {same_full}")
    print("Note: load_full uses weights_only=False — only open files you trust.")

    print(f"\nDone. Artifacts left in {tmpdir} (delete at will).")


if __name__ == "__main__":
    main()
