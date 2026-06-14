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


def build_model() -> ESNModel:
    """The architecture must be re-created before loading weights into it."""
    torch.manual_seed(0)  # seed only affects the initial (pre-load) weights
    return rd.classic_esn(reservoir_size=200, feedback_size=3, output_size=3)


def main() -> None:
    torch.manual_seed(42)
    tmpdir = Path(tempfile.mkdtemp(prefix="resdag_example_"))
    print(f"Writing artifacts to {tmpdir}\n")

    # A small trained model to persist
    series = torch.cumsum(0.05 * torch.randn(1, 1001, 3), dim=1)  # (1, 1001, 3)
    model = build_model()
    ESNTrainer(model).fit(
        warmup_inputs=(series[:, :200],),
        train_inputs=(series[:, 200:800],),
        targets={"output": series[:, 201:801]},
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

    restored = build_model()  # fresh instance, different random weights
    restored.load(path)  # now identical to `model`

    x = torch.randn(2, 50, 3)  # (batch, time, features)
    model.reset_reservoirs()
    restored.reset_reservoirs()
    same = torch.allclose(model(x), restored(x))
    print(f"restored model reproduces original outputs: {same}")

    # ------------------------------------------------------------------
    # 2. Checkpoints: metadata + reservoir states
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Checkpoints with metadata and reservoir states")
    print("=" * 70)

    model.reset_reservoirs()
    model.warmup(series[:, :200])  # put something in the reservoir state

    ckpt = tmpdir / "checkpoint.pt"
    model.save(ckpt, include_states=True, epoch=10, val_mse=0.0123, note="demo run")
    print("saved with include_states=True and metadata kwargs")

    # Metadata is plain torch.save payload — inspect without a model:
    payload = torch.load(ckpt, weights_only=False)
    print(f"metadata: {payload['metadata']}")
    print(f"stored reservoir states: {list(payload['reservoir_states'].keys())}")

    # ------------------------------------------------------------------
    # 3. load_from_file class method
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. ESNModel.load_from_file")
    print("=" * 70)

    restored = ESNModel.load_from_file(ckpt, model=build_model(), load_states=True)
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

    f_warmup = series[:, 600:800]

    # Reference: forecast in one go
    model.reset_reservoirs()
    reference = model.forecast(f_warmup, horizon=100)

    # Same forecast, split across a save/load boundary. Warm up on all but
    # the last sample, save, and hand the remaining sample to the resumed
    # forecast — it completes the warmup and continues autoregressively.
    model.reset_reservoirs()
    model.warmup(f_warmup[:, :-1])
    model.save(tmpdir / "warmed_up.pt", include_states=True)

    resumed = ESNModel.load_from_file(
        tmpdir / "warmed_up.pt", model=build_model(), load_states=True
    )
    continued = resumed.forecast(f_warmup[:, -1:], horizon=100, reset=False)

    dev = (reference - continued).abs().amax(dim=(0, 2))  # per-step max deviation
    print(f"max deviation at step 0:  {dev[0].item():.1e}")
    print(f"max deviation at step 20: {dev[20].item():.1e}")
    print(
        f"first 20 steps match (atol=1e-3): "
        f"{torch.allclose(reference[:, :20], continued[:, :20], atol=1e-3)}"
    )
    print("The restored state matches to float precision (~1e-7); as with any")
    print("re-run of a chaotic simulation, rounding differences amplify over")
    print("long autoregressive horizons.")

    # ------------------------------------------------------------------
    # 5. Cross-device
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Cross-device save/load")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_model = build_model().to("cuda")
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

    # No build_model() call — the architecture is reconstructed from the file.
    reloaded, meta = ESNModel.load_full(full_path, return_metadata=True)
    print(f"reconstructed without rebuilding; metadata: {meta}")

    model.reset_reservoirs()
    reloaded.reset_reservoirs()
    same_full = torch.allclose(model(x), reloaded(x))
    print(f"reloaded full model reproduces original outputs: {same_full}")
    print("Note: load_full uses weights_only=False — only open files you trust.")

    print(f"\nDone. Artifacts left in {tmpdir} (delete at will).")


if __name__ == "__main__":
    main()
