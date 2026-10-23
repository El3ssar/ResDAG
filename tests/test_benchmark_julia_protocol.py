"""Opt-in oracles against the pinned, real Julia comparison implementation.

Run with ``RESDAG_TEST_JULIA=1`` and ``julia`` on PATH after instantiating
``benchmarks/julia``. ``RESDAG_JULIA_PROJECT`` can select an isolated copy of
that pinned project. Ordinary test runs need neither Julia nor its packages.
"""

import base64
import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.rc_bench.adapters import julia_adapter
from benchmarks.rc_bench.config import HParams, Point
from benchmarks.rc_bench.data import lorenz, prepare_point


def test_julia_budget_greeting_preserves_verified_blas_configuration():
    from benchmarks.rc_bench.budget_adapters import _parse_julia_greeting

    config = "LBTConfig([ILP64] libopenblas64_.dylib)\npath\twith separators"
    versions, threads = _parse_julia_greeting(
        ["READY", "1.12.5", "0.12.23", "1", "1", "lbt", base64.b64encode(config.encode()).decode()]
    )
    assert versions == {"julia": "1.12.5", "ReservoirComputing": "0.12.23"}
    assert threads == {
        "julia_threads": 1,
        "blas_threads": 1,
        "blas_vendor": "lbt",
        "blas_config": config,
    }


@pytest.mark.parametrize(
    "index,value",
    [(0, "OLD"), (1, ""), (2, ""), (3, "2"), (4, "2"), (4, "unknown"), (5, ""), (6, "?"), (6, "")],
)
def test_julia_budget_greeting_rejects_missing_or_unverified_threads(index, value):
    from benchmarks.rc_bench.budget_adapters import _parse_julia_greeting

    fields = ["READY", "1.12.5", "0.12.23", "1", "1", "lbt", "T3BlbkJMQVM="]
    fields[index] = value
    with pytest.raises(RuntimeError):
        _parse_julia_greeting(fields)
    with pytest.raises(RuntimeError):
        _parse_julia_greeting(fields[:3])


@pytest.fixture(scope="module")
def real_julia(tmp_path_factory):
    if os.environ.get("RESDAG_TEST_JULIA") != "1":
        pytest.skip("set RESDAG_TEST_JULIA=1 to run the real pinned Julia protocol")
    executable = shutil.which("julia")
    if executable is None:
        pytest.fail("RESDAG_TEST_JULIA=1 requires julia on PATH")

    project = Path(os.environ.get("RESDAG_JULIA_PROJECT", julia_adapter._JULIA_PROJECT))
    project = project.resolve()
    manifest = tomllib.loads((project / "Manifest.toml").read_text())
    version = manifest["deps"]["ReservoirComputing"][0]["version"]
    worker = Path(julia_adapter._JULIA_SCRIPT).resolve()
    directory = tmp_path_factory.mktemp("real-julia-protocol")

    point = Point(reservoir_size=20, train_len=45, warmup=3, horizon=4)
    hp = HParams(connectivity=0.4)
    series, _ = prepare_point(lorenz(point.series_len), point)
    np.ascontiguousarray(series, dtype="<f8").tofile(directory / "data.bin")
    oracle = directory / "export.jl"
    oracle.write_text(
        f"""include({json.dumps(str(worker))})
cfg=(dim={hp.dim}, reservoir_size={point.reservoir_size},
     train_len={point.train_len}, warmup={point.warmup}, horizon={point.horizon},
     seed={hp.seed}, lr={hp.leak_rate}, sr={hp.spectral_radius},
     connectivity={hp.connectivity}, input_scaling={hp.input_scaling}, ridge={hp.ridge})
@assert string(pkgversion(ReservoirComputing)) == {json.dumps(version)}
data=load_data(joinpath(@__DIR__, "data.bin"), cfg.dim, {point.series_len})
esn, ps, st=build_esn(cfg)
@assert eltype(ps.reservoir.input_matrix) == Float64
@assert eltype(ps.reservoir.reservoir_matrix) == Float64
write(joinpath(@__DIR__, "Win.bin"), ps.reservoir.input_matrix)
write(joinpath(@__DIR__, "W.bin"), ps.reservoir.reservoir_matrix)
ps, _=do_train!(esn, ps, st, data, cfg)
@assert eltype(ps.readout.weight) == Float64
write(joinpath(@__DIR__, "Wout.bin"), ps.readout.weight)
prediction=do_forecast(esn, ps, st, data, cfg)
@assert eltype(prediction) == Float64
write(joinpath(@__DIR__, "prediction.bin"), prediction)
"""
    )
    environment = os.environ | {
        "JULIA_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "JULIA_PKG_PRECOMPILE_AUTO": "0",
    }
    process = subprocess.run(
        [executable, f"--project={project}", "--startup-file=no", str(oracle)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
        check=False,
    )
    assert process.returncode == 0, process.stderr

    def matrix(name, shape):
        return np.fromfile(directory / name, dtype="<f8").reshape(shape, order="F")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(julia_adapter, "_JULIA_PROJECT", project)
        patch.setenv("JULIA_NUM_THREADS", "1")
        patch.setenv("OPENBLAS_NUM_THREADS", "1")
        patch.setenv("JULIA_PKG_PRECOMPILE_AUTO", "0")
        yield {
            "adapter": julia_adapter.ReservoirComputingJLAdapter(),
            "series": series,
            "point": point,
            "hp": hp,
            "win": matrix("Win.bin", (point.reservoir_size, hp.dim)),
            "w": matrix("W.bin", (point.reservoir_size, point.reservoir_size)),
            "wout": matrix("Wout.bin", (hp.dim, point.reservoir_size)),
            "prediction": matrix("prediction.bin", (hp.dim, point.horizon)).T,
            "project": project,
            "directory": directory,
        }


def _advance(state, sample, data):
    rate = data["hp"].leak_rate
    return (1 - rate) * state + rate * np.tanh(data["win"] @ sample + data["w"] @ state)


def test_real_julia_fit_matches_independent_augmented_ridge(real_julia):
    data = real_julia
    point, hp, series = data["point"], data["hp"], data["series"]
    state = np.zeros(point.reservoir_size)
    states = []
    for sample in series[:45]:
        state = _advance(state, sample, data)
        states.append(state)
    features = np.asarray(states)[3:45]
    labels = series[4:46]
    # This independently admits x[0:45] and y[1:46], then washes out three
    # states/labels. A constant/bias column would fail the weight-shape oracle.
    augmented = np.vstack([features, np.sqrt(hp.ridge) * np.eye(point.reservoir_size)])
    targets = np.vstack([labels, np.zeros((point.reservoir_size, hp.dim))])
    expected, *_ = np.linalg.lstsq(augmented, targets, rcond=None)
    np.testing.assert_allclose(data["wout"], expected.T, rtol=1e-7, atol=1e-8)


def test_real_julia_first_unseen_output_and_full_recurrence(real_julia):
    data = real_julia
    state = np.zeros(data["point"].reservoir_size)
    # Last observed sample is 45; synchronizing on 43,44,45 already emits the
    # prediction for unseen sample 46. An extra call would skip that first step.
    for sample in data["series"][43:46]:
        state = _advance(state, sample, data)
    expected = [data["wout"] @ state]
    for _ in range(3):
        state = _advance(state, expected[-1], data)
        expected.append(data["wout"] @ state)
    np.testing.assert_allclose(data["prediction"], expected, rtol=1e-11, atol=1e-11)
    actual = data["adapter"].predict_trajectory(data["series"], data["hp"], data["point"])
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, data["prediction"])


def test_real_julia_future_truth_cannot_change_forecast(real_julia):
    data = real_julia
    changed = data["series"].copy()
    changed[46:] = 1e9
    actual = data["adapter"].predict_trajectory(changed, data["hp"], data["point"])
    np.testing.assert_array_equal(actual, data["prediction"])


def test_real_julia_requested_repeats_reset_state_and_score_unseen_truth(real_julia):
    data = real_julia
    adapter, series, hp, point = data["adapter"], data["series"], data["hp"], data["point"]
    train = adapter.time_train(series, hp, point, repeats=2, warmups=1)
    forecast = adapter.time_forecast(series, hp, point, repeats=2, warmups=1)
    single = adapter.time_forecast(series, hp, point, repeats=1, warmups=0)
    assert len(train.times) == len(forecast.times) == 2
    assert len(single.times) == 1
    assert all(np.isfinite(value) and value > 0 for value in train.times + forecast.times)
    expected = np.sqrt(np.mean((data["prediction"] - series[46:50]) ** 2))
    assert forecast.rmse == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert single.rmse == pytest.approx(forecast.rmse, rel=0, abs=0)


def test_real_julia_later_origin_keeps_fit_fixed_and_starts_at_unseen_sample(real_julia):
    data = real_julia
    point = replace(data["point"], origin=48, horizon=2)
    state = np.zeros(point.reservoir_size)
    for sample in data["series"][45:48]:
        state = _advance(state, sample, data)
    expected = [data["wout"] @ state]
    state = _advance(state, expected[-1], data)
    expected.append(data["wout"] @ state)
    actual = data["adapter"].predict_trajectory(data["series"], data["hp"], point)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    timing = data["adapter"].time_forecast(data["series"], data["hp"], point, 1, 0)
    assert timing.rmse == pytest.approx(np.sqrt(np.mean((actual - data["series"][48:50]) ** 2)))


def test_real_julia_budget_worker_rejects_multiple_julia_threads_before_ready(real_julia):
    worker = Path(__file__).resolve().parents[1] / "benchmarks/julia/rc_budget.jl"
    result = subprocess.run(
        [
            shutil.which("julia"),
            f"--project={real_julia['project']}",
            "--startup-file=no",
            str(worker),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ | {"JULIA_NUM_THREADS": "2"},
        check=False,
    )
    assert result.returncode != 0
    assert not result.stdout.startswith("READY")
    assert "requires one Julia thread" in result.stderr


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_real_persistent_julia_common_recipe_matches_numpy_fit_and_recurrence(real_julia, storage):
    from benchmarks.rc_bench.budget_adapters import CommonParams, JuliaBudgetAdapter

    data = real_julia
    directory = data["directory"] / storage
    directory.mkdir()
    np.ascontiguousarray(data["series"][:46], dtype="<f8").tofile(directory / "data.bin")
    worker = Path(__file__).resolve().parents[1] / "benchmarks/julia/rc_budget.jl"
    script = directory / "oracle.jl"
    script.write_text(
        f"""include({json.dumps(str(worker))})
esn, ps, st=budget_model(3,20,.9,.3,.4,.2,.1,1234; sparse={str(storage == 'sparse').lower()})
@assert eltype(ps.reservoir.input_matrix) == Float64
@assert eltype(ps.reservoir.bias) == Float64
@assert eltype(ps.reservoir.reservoir_matrix) == Float64
write(joinpath(@__DIR__,"Win.bin"),ps.reservoir.input_matrix)
write(joinpath(@__DIR__,"W.bin"),Matrix(ps.reservoir.reservoir_matrix))
write(joinpath(@__DIR__,"bias.bin"),ps.reservoir.bias)
values=load_data(joinpath(@__DIR__,"data.bin"),3,46)
ps, _=do_train!(esn,ps,st,values,(train_len=45,warmup=3,ridge=1e-6))
write(joinpath(@__DIR__,"Wout.bin"),ps.readout.weight)
"""
    )
    result = subprocess.run(
        [shutil.which("julia"), f"--project={data['project']}", "--startup-file=no", str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    win = np.fromfile(directory / "Win.bin").reshape(20, 3, order="F")
    w = np.fromfile(directory / "W.bin").reshape(20, 20, order="F")
    bias = np.fromfile(directory / "bias.bin")
    wout = np.fromfile(directory / "Wout.bin").reshape(3, 20, order="F")
    assert np.ptp(bias) > 0 and np.max(np.abs(bias)) <= 0.1
    assert np.count_nonzero(win) == win.size and np.max(np.abs(win)) <= 0.2
    assert np.count_nonzero(w) == 20 * 8  # pinned Julia mask keeps eight entries/column

    def advance(state, sample):
        return 0.7 * state + 0.3 * np.tanh(win @ sample + w @ state + bias)

    state = np.zeros(20)
    states = []
    for sample in data["series"][:45]:
        state = advance(state, sample)
        states.append(state)
    design = np.vstack([np.asarray(states)[3:], 1e-3 * np.eye(20)])
    labels = np.vstack([data["series"][4:46], np.zeros((20, 3))])
    expected = np.linalg.lstsq(design, labels, rcond=None)[0].T
    np.testing.assert_allclose(wout, expected, rtol=1e-7, atol=1e-8)
    adapter = JuliaBudgetAdapter(project=data["project"], storage=storage)
    try:
        assert adapter.thread_metadata["julia_threads"] == 1
        assert adapter.thread_metadata["blas_threads"] == 1
        assert adapter.thread_metadata["blas_vendor"]
        assert adapter.thread_metadata["blas_config"]
        candidate, timing = adapter.fit(
            data["series"][:46], CommonParams(20, 0.9, 0.3, 0.2, 0.1, 8, 1e-6), 1234, 3
        )
        assert timing["fit_calls"] == 1 and timing["parameter_tree_heap_bytes"] > 0
        for history in (data["series"][43:46], data["series"][45:48]):
            state = np.zeros(20)
            for sample in history:
                state = advance(state, sample)
            expected = [wout @ state]
            for _ in range(3):
                state = advance(state, expected[-1])
                expected.append(wout @ state)
            actual, _ = candidate.forecast(history, 4)
            repeated, _ = candidate.forecast(history, 4)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(actual, repeated)
    finally:
        adapter.close()
