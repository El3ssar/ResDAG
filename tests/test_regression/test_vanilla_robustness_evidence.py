"""Reconstruct the published vanilla study with independent stdlib-only oracles."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import re
import statistics
import struct
import zlib
from functools import lru_cache
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/assets/data/phase-five-vanilla-robustness-2026-10-24.json"
PREDECESSOR = ROOT / "docs/assets/data/phase-five-independent-trajectories-2026-10-23.json"
RUNTIME_COMMIT = "c2a97583aee1c30c3dcd791c69aafb9168096fc3"
SYSTEMS = {"lorenz", "rossler"}
THRESHOLDS = ("0.3", "0.5")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _identity(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _digest(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _seconds(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value >= 0


@lru_cache(maxsize=1)
def _publication():
    return json.loads(PUBLICATION.read_text())


def _errors(row, horizon):
    trace = row["error_trace"]
    assert trace["encoding"] == "zlib-base64 little-endian float64"
    assert trace["length"] == horizon
    payload = zlib.decompress(base64.b64decode(trace["data"], validate=True))
    assert len(payload) == 8 * horizon
    assert _sha(payload) == trace["sha256"]
    values = struct.unpack(f"<{horizon}d", payload)
    assert all(math.isnan(value) or value >= 0 for value in values)
    return values


def _scores(row, config):
    errors = _errors(row, config["horizon"])
    reconstructed = {}
    assert type(row["finite"]) is bool
    if all(math.isfinite(value) for value in errors):
        assert row["finite"] is True
    for threshold in THRESHOLDS:
        crossing = next(
            (
                index
                for index, error in enumerate(errors)
                if not math.isfinite(error) or error >= float(threshold)
            ),
            len(errors),
        )
        expected = {
            "steps": crossing,
            "lyapunov_times": crossing * config["dt"] * config["lambda_max"],
            "right_censored": crossing == len(errors),
        }
        assert type(row["scores"][threshold]["steps"]) is int
        assert type(row["scores"][threshold]["right_censored"]) is bool
        assert row["scores"][threshold] == expected
        reconstructed[threshold] = expected
    assert set(row["scores"]) == set(THRESHOLDS)
    return reconstructed


def _summary(rows, reconstructed):
    result = {}
    trajectory_ids = sorted({row["trajectory"] for row in rows})
    for threshold in THRESHOLDS:
        values = [scores[threshold]["lyapunov_times"] for scores in reconstructed]
        medians = [
            statistics.median(
                scores[threshold]["lyapunov_times"]
                for row, scores in zip(rows, reconstructed)
                if row["trajectory"] == trajectory
            )
            for trajectory in trajectory_ids
        ]
        result[threshold] = {
            "median_of_trajectory_medians": statistics.median(medians),
            "trajectory_medians": medians,
            "individual_range": [min(values), max(values)],
            "right_censored": sum(scores[threshold]["right_censored"] for scores in reconstructed),
            "forecasts": len(rows),
        }
    return result


def _evaluation(evaluation, config, seeds, trajectory_count, complete):
    expected_fits = [(trajectory, seed) for trajectory in range(trajectory_count) for seed in seeds]
    expected_rows = [
        (*fit, origin) for fit in expected_fits for origin in range(config["origin_count"])
    ]
    fits, rows = evaluation["fits"], evaluation["forecasts"]
    fit_keys = [(fit["trajectory"], fit["seed"]) for fit in fits]
    row_keys = [(row["trajectory"], row["seed"], row["origin"]) for row in rows]
    assert fit_keys == expected_fits[: len(fits)]
    assert row_keys == expected_rows[: len(rows)]
    assert len(fits) <= len(expected_fits) and len(rows) <= len(expected_rows)
    assert all(key[:2] in fit_keys for key in row_keys)
    for fit in fits:
        _seconds(fit["seconds"])
        _digest(fit["reservoir_sha256"])
    for seed in seeds:
        assert len({fit["reservoir_sha256"] for fit in fits if fit["seed"] == seed}) <= 1
    reconstructed = []
    for row in rows:
        assert (
            row["origin_index"]
            == row["history_observations"]
            == (config["train_observations"] + row["origin"] * config["horizon"])
        )
        _seconds(row["seconds"])
        _digest(row["prediction_sha256"])
        reconstructed.append(_scores(row, config))
    scales = evaluation["training_scales"]
    assert [item["trajectory"] for item in scales] == list(range(len(scales)))
    assert len(scales) <= trajectory_count
    for item in scales:
        assert len(item["scale"]) == 3
        assert all(math.isfinite(value) and value > 0 for value in item["scale"])
    for warning in evaluation["warnings"]:
        assert all(isinstance(warning[key], str) for key in ("category", "message", "file"))
        assert type(warning["line"]) is int and warning["line"] > 0
    if complete:
        assert fit_keys == expected_fits and row_keys == expected_rows
        assert len(scales) == trajectory_count
        assert "in_progress" not in evaluation and "error" not in evaluation
        assert evaluation["summary"] == _summary(rows, reconstructed)
    else:
        assert "summary" not in evaluation
        operations = []
        for trajectory, seed in expected_fits:
            operations.append({"operation": "fit", "trajectory": trajectory, "seed": seed})
            operations.extend(
                {"operation": "forecast", "trajectory": trajectory, "seed": seed, "origin": origin}
                for origin in range(config["origin_count"])
            )
        completed_operations = len(fits) + len(rows)
        assert completed_operations < len(operations)
        assert evaluation["in_progress"] == operations[completed_operations]


def _params(params, protocol):
    assert set(params) == {
        "reservoir_size",
        "spectral_radius",
        "leak_rate",
        "input_scaling",
        "bias_scaling",
        "alpha",
        "mean_degree",
        "input_to_readout",
    }
    assert params["reservoir_size"] in protocol["widths"]
    assert params["mean_degree"] in protocol["other_ranges"]["mean_degree"]
    assert type(params["input_to_readout"]) is bool
    for key, bounds in (
        ("spectral_radius", "spectral_radius"),
        ("leak_rate", "leak_rate"),
        ("input_scaling", "input_scaling_log"),
        ("bias_scaling", "bias_scaling_log"),
        ("alpha", "alpha_log"),
    ):
        lower, upper = protocol["other_ranges"][bounds]
        assert math.isfinite(params[key]) and lower <= params[key] <= upper


def _datasets(rows, phase, system, count):
    assert len(rows) == count
    assert len({item["sha256"] for item in rows}) == count
    for index, item in enumerate(rows):
        assert item["path"] == f"{phase}/{system}-{index}.npy"
        _digest(item["sha256"])
        _seconds(item["seconds"])


def test_vanilla_protocol_source_and_admitted_scripts_are_bound():
    publication = _publication()
    registered = publication["preregistration"]
    protocol, results = registered["protocol"], publication["results"]
    assert _sha(_file_bytes(protocol)) == registered["sha256"] == results["plan_sha256"]
    frozen = publication["frozen_winners"]
    assert frozen["plan_sha256"] == registered["sha256"]
    assert _sha(_file_bytes(frozen)) == results["winners_sha256"]
    assert frozen["winners"] == results["winners"]
    assert frozen["test_used_for_selection"] is False
    source = publication["source"]
    assert source["runtime_commit"] == RUNTIME_COMMIT
    # Admission preceded committing the already-frozen module bytes. Keep its
    # base revision intact; the complete source manifest binds the later commit.
    assert protocol["commit"] == "ca40fe4b9806a97ec11813a791b6f1cb11c58d0f"
    files = source["files"]
    assert files == protocol["source_files"]
    assert _identity(files) == protocol["source_manifest_sha256"]
    assert len(files) == 206 and sum(path.endswith(".py") for path in files) == 205
    assert "src/resdag/py.typed" in files
    for path, digest in files.items():
        relative = PurePosixPath(path)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert path.startswith("src/resdag/")
        _digest(digest)
    assert set(publication["scripts"]) == {"runner", "thread_policy"}
    for name, key in (("runner", "runner_sha256"), ("thread_policy", "thread_helper_sha256")):
        script = publication["scripts"][name]
        assert _sha(script["source"].encode()) == script["sha256"] == protocol[key]
        compile(script["source"], name, "exec")
    assert results["source_imports_verified"] is True
    assert results["scope"] == (
        "recipe robustness; timing observations are not a controlled speed ranking"
    )


def test_vanilla_independent_design_and_bounded_search_are_preregistered():
    protocol = _publication()["preregistration"]["protocol"]
    assert protocol["kind"] == "independent-trajectory recipe search"
    assert set(protocol["systems"]) == SYSTEMS
    assert protocol["trials_per_system"] == 24
    assert protocol["search_timeout_seconds"] == 600
    assert protocol["search_sampler_seed"] == 2026102411
    assert protocol["initial_condition_rng_seed"] == 2026102407
    assert protocol["validation_seeds"] == [6849201, 6849202, 6849203]
    assert protocol["final_seeds"] == [7849201, 7849202, 7849203]
    assert set(protocol["validation_seeds"]).isdisjoint(protocol["final_seeds"])
    assert protocol["input_to_readout_choices"] == [True, False]
    assert protocol["widths"] == [300, 400, 600, 1000]
    assert protocol["other_ranges"] == {
        "spectral_radius": [0.05, 1.4],
        "leak_rate": [0.1, 1.0],
        "input_scaling_log": [0.01, 1.0],
        "bias_scaling_log": [0.01, 0.5],
        "alpha_log": [1e-12, 1e-3],
        "mean_degree": [3, 6, 12],
    }
    assert protocol["thresholds"] == [0.3, 0.5]
    assert protocol["pruning"] is False
    assert protocol["objective"] == "median of per-trajectory medians at 0.3"
    assert protocol["solver"] == "Torch SVD, float64, unpenalized intercept"
    assert protocol["metric_scale_rows"] == (
        "raw[washout + 1 : train_observations], matching preceding primary protocol"
    )
    assert protocol["solver_parameters"] == {"method": "dop853", "rtol": 1e-10, "atol": 1e-12}
    all_conditions = []
    expected = {"lorenz": (0.02, 0.9056, 10001, 2000), "rossler": (0.145, 0.0714, 8501, 1000)}
    for system, config in protocol["systems"].items():
        assert (
            config["dt"],
            config["lambda_max"],
            config["train_observations"],
            config["discard"],
        ) == expected[system]
        assert config["washout"] == 500 and config["origin_count"] == 3
        assert config["parameters"] == (
            {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
            if system == "lorenz"
            else {"a": 0.2, "b": 0.2, "c": 5.7}
        )
        assert config["horizon"] == math.ceil(10 / (config["dt"] * config["lambda_max"]))
        assert len(config["validation_ics"]) == 3 and len(config["final_ics"]) == 5
        for condition in config["validation_ics"] + config["final_ics"]:
            assert len(condition) == 3
            assert all(math.isfinite(value) and 0.5 <= value <= 1.5 for value in condition)
            all_conditions.append(tuple(condition))
    assert len(all_conditions) == len(set(all_conditions)) == 16
    previous = json.loads(PREDECESSOR.read_text())["preregistration"]["protocol"]
    assert set(protocol["validation_seeds"] + protocol["final_seeds"]).isdisjoint(
        previous["initialization_seeds"]
    )
    assert set(all_conditions).isdisjoint(
        tuple(item["initial_condition"]) for item in previous["trajectories"]
    )


def test_vanilla_incumbents_and_controlled_baselines_use_the_same_reservoirs():
    publication = _publication()
    protocol, results = publication["preregistration"]["protocol"], publication["results"]
    assert protocol["incumbent_provenance"] == {
        "file": PREDECESSOR.name,
        "sha256": _sha(PREDECESSOR.read_bytes()),
    }
    old = json.loads(PREDECESSOR.read_text())["preregistration"]["protocol"]["recipes"]
    for system in SYSTEMS:
        recipe = old[f"{system}-classic"]
        assert recipe["family"] == "classic" and recipe["panel"] == "primary-capability"
        params = recipe["params"]
        expected = {
            "reservoir_size": params["width"],
            "spectral_radius": params["rho"],
            "leak_rate": params["leak"],
            "input_scaling": params["input_scale"],
            "bias_scaling": params["bias_scale"],
            "alpha": params["alpha"],
            "mean_degree": params["degree"],
            "input_to_readout": True,
        }
        assert protocol["incumbents"][system] == expected
        validation = results["validation"][system]
        left, right = validation["trials"][:2]
        assert [left["trial"], right["trial"]] == [0, 1]
        assert left["params"] == expected
        assert right["params"] == {**expected, "input_to_readout": False}
        assert validation["states"]["0"] == validation["states"]["1"] == "COMPLETE"
        assert left["status"] == right["status"] == "complete"

        def fit_keys(record):
            return [
                (fit["trajectory"], fit["seed"], fit["reservoir_sha256"])
                for fit in record["evaluation"]["fits"]
            ]

        assert len(fit_keys(left)) == 9 and fit_keys(left) == fit_keys(right)
        assert validation["controlled_feature_comparison"] == {
            "input_plus_state": left["evaluation"]["summary"],
            "state_only": right["evaluation"]["summary"],
            "same_reservoir_weights": True,
            "changed_setting": "input_to_readout only",
        }


def test_vanilla_every_validation_cell_and_failure_receipt_is_retained():
    publication = _publication()
    protocol, results = publication["preregistration"]["protocol"], publication["results"]
    assert set(results["validation"]) == set(results["winners"]) == SYSTEMS
    for system in SYSTEMS:
        validation = results["validation"][system]
        _datasets(validation["datasets"], "validation", system, 3)
        records, states = validation["trials"], validation["states"]
        assert 2 <= len(records) <= protocol["trials_per_system"]
        assert [record["trial"] for record in records] == list(range(len(records)))
        assert set(states) == {str(record["trial"]) for record in records}
        _seconds(validation["search_seconds"])
        for record in records:
            _params(record["params"], protocol)
            assert states[str(record["trial"])] in {"COMPLETE", "FAIL"}
            complete = states[str(record["trial"])] == "COMPLETE"
            assert record["status"] == ("complete" if complete else "failed")
            if complete:
                assert "error" not in record
            else:
                assert isinstance(record["error"], str) and record["error"]
                error_type, separator, message = record["error"].partition(": ")
                assert error_type.isidentifier() and separator and message
                assert "Dataset changed after registration" not in record["error"]
                assert (
                    "Readout fitting unexpectedly changed reservoir weights" not in record["error"]
                )
            _evaluation(
                record["evaluation"],
                protocol["systems"][system],
                protocol["validation_seeds"],
                3,
                complete,
            )
            scales = record["evaluation"]["training_scales"]
            assert scales == records[0]["evaluation"]["training_scales"][: len(scales)]
        completed = [record for record in records if record["status"] == "complete"]
        selected = min(
            completed,
            key=lambda record: (
                -record["evaluation"]["summary"]["0.3"]["median_of_trajectory_medians"],
                record["trial"],
            ),
        )
        assert results["winners"][system] == {
            "trial": selected["trial"],
            "params": selected["params"],
            "validation_score": selected["evaluation"]["summary"]["0.3"][
                "median_of_trajectory_medians"
            ],
        }


def test_vanilla_all_final_fits_forecasts_traces_and_nested_summaries_are_complete():
    publication = _publication()
    protocol, results = publication["preregistration"]["protocol"], publication["results"]
    assert set(results["final"]) == SYSTEMS
    fit_count = forecast_count = 0
    validation_hashes, final_hashes = set(), set()
    for system in SYSTEMS:
        final = results["final"][system]
        assert set(final) == {"datasets", "incumbent", "winner"}
        _datasets(final["datasets"], "final", system, 5)
        validation_hashes.update(
            item["sha256"] for item in results["validation"][system]["datasets"]
        )
        final_hashes.update(item["sha256"] for item in final["datasets"])
        for variant in ("incumbent", "winner"):
            evaluation = final[variant]
            _evaluation(evaluation, protocol["systems"][system], protocol["final_seeds"], 5, True)
            fit_count += len(evaluation["fits"])
            forecast_count += len(evaluation["forecasts"])
        assert final["incumbent"]["training_scales"] == final["winner"]["training_scales"]
    assert len(validation_hashes) == 6 and len(final_hashes) == 10
    assert validation_hashes.isdisjoint(final_hashes)
    assert fit_count == 60 and forecast_count == 180


def test_vanilla_admitted_runner_binds_final_parameters_and_canonical_model_seeds():
    tree = ast.parse(_publication()["scripts"]["runner"]["source"])
    expected_loop = ast.parse(
        'for variant, params in (("incumbent", incumbents[name]), '
        '("winner", results["winners"][name]["params"])):\n    pass\n'
    ).body[0]
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    final_loops = [
        node
        for node in loops
        if ast.dump(node.target) == ast.dump(expected_loop.target)
        and ast.dump(node.iter) == ast.dump(expected_loop.iter)
    ]
    assert len(final_loops) == 1
    final_calls = [
        node
        for node in ast.walk(final_loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_evaluate"
    ]
    assert len(final_calls) == 1
    expected_call = ast.parse(
        '_evaluate(params, datasets, config, plan["final_seeds"], rd, torch, np, evaluation)',
        mode="eval",
    ).body
    assert ast.dump(final_calls[0]) == ast.dump(expected_call)
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.dump(node.func) == ast.dump(ast.parse("rd.ESN", mode="eval").body)
    ]
    assert len(constructors) == 1
    kwargs = {item.arg: item.value for item in constructors[0].keywords}
    for key, expression in {
        "input_size": "3",
        "washout": 'config["washout"]',
        "readout": '"svd"',
        "readout_bias": "True",
        "seed": "seed",
        "dtype": "torch.float64",
        None: "settings",
    }.items():
        assert ast.dump(kwargs[key]) == ast.dump(ast.parse(expression, mode="eval").body)
    settings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "settings" for target in node.targets)
    ]
    expected_settings = ast.parse(
        "{key: value for key, value in params.items() "
        'if key not in ("mean_degree", "input_scaling")}',
        mode="eval",
    ).body
    assert len(settings) == 1 and ast.dump(settings[0]) == ast.dump(expected_settings)
    evaluator_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_evaluate"
        and len(node.args) >= 4
    ]
    assert {ast.unparse(call.args[3]) for call in evaluator_calls} == {
        "plan['validation_seeds']",
        "plan['final_seeds']",
    }
    final_generation = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_generate"
        and len(node.args) >= 4
        and isinstance(node.args[3], ast.Name)
        and node.args[3].id == "final"
    ]
    freeze_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write"
        and len(node.args) == 2
        and ast.unparse(node.args[0]) == "args.output / 'frozen-winners.json'"
    ]
    assert len(final_generation) == len(freeze_calls) == 1
    assert freeze_calls[0].lineno < final_generation[0].lineno
    winner_targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "results['winners'][name]" for target in node.targets)
    ]
    assert len(winner_targets) == 1 and winner_targets[0].lineno < freeze_calls[0].lineno


def test_vanilla_recorded_native_policy_stays_within_the_declared_execution_scope():
    policy = _publication()["results"]["native_policy"]
    assert policy["workers"] == 1 and policy["dtype"] == "float64"
    assert policy["fresh_cli_bootstrap"] is True
    assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
    assert set(policy["requested_environment"]) == {
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "JULIA_NUM_THREADS",
    }
    assert set(policy["requested_environment"].values()) == {"1"}
    assert policy["native_pools"]
    assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
    if policy["accelerate"] is not None:
        assert policy["accelerate"]["BLASGetThreading"] == 1
        assert policy["accelerate"]["scope"] == "calling thread"
    assert "not an OS/kernel thread-occupancy measurement" in policy["verification_boundary"]


def test_vanilla_integrity_receipt_accounts_for_every_dataset_source_and_model_failure():
    publication = _publication()
    results, integrity = publication["results"], publication["integrity"]
    assert (
        integrity["dataset_files_verified"]
        == sum(
            len(results[phase][system]["datasets"])
            for phase in ("validation", "final")
            for system in SYSTEMS
        )
        == 16
    )
    assert integrity["source_files_verified"] == len(publication["source"]["files"]) == 206
    assert integrity["source_matches_runtime_commit"] is True
    assert integrity["observed_model_failures"] == sum(
        trial["status"] == "failed"
        for validation in results["validation"].values()
        for trial in validation["trials"]
    )
    assert integrity["observed_integrity_failures"] == 0
    assert integrity["report_sha256"] == _sha(_file_bytes(results))
