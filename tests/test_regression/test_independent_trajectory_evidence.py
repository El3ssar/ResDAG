"""Independently reconstruct the preregistered accuracy evidence using stdlib only."""

import ast
import base64
import hashlib
import json
import math
import random
import statistics
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/assets/data/phase-five-independent-trajectories-2026-10-23.json"


def _report():
    return json.loads(PUBLICATION.read_text())


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _errors(row):
    trace = row["normalized_error_trace"]
    assert trace["id"] == row["id"]
    assert trace["encoding"] == "zlib-base64-float64-le"
    data = zlib.decompress(base64.b64decode(trace["data"], validate=True))
    assert _sha(data) == trace["sha256"]
    assert trace["count"] == row["horizon"]
    assert len(data) == 8 * trace["count"]
    return struct.unpack(f"<{trace['count']}d", data)


def test_independent_study_preregistration_source_and_helper_bindings():
    report = _report()
    registered = report["preregistration"]
    protocol = registered["protocol"]
    assert _sha((json.dumps(protocol, indent=2, allow_nan=False) + "\n").encode()) == (
        registered["sha256"]
    )
    assert registered["receipt"]["sha256"] == registered["sha256"]
    assert registered["receipt"]["data_files_created"] is False
    assert registered["receipt"]["numerical_modules_imported"] == []
    assert _canonical(protocol["recipes"]) == protocol["recipe_manifest_sha256"]
    assert _canonical(protocol["source_manifest"]) == protocol["source_manifest_sha256"]
    assert _canonical(protocol["runner_manifest"]) == protocol["runner_manifest_sha256"]
    source = report["source"]
    assert source["files"] == protocol["source_manifest"]
    assert source["manifest_sha256"] == protocol["source_manifest_sha256"]
    assert source["file_count"] == len(source["files"]) == 206
    assert source["python_module_count"] == sum(p.endswith(".py") for p in source["files"]) == 205
    assert source["archive_base_commit"] == "3b34005efd93a2f6573ecd96c9a5b23daecdf547"
    for name, overlay in source["overlays"].items():
        assert _sha(overlay["source"].encode()) == overlay["sha256"] == source["files"][name]
    assert set(report["scripts"]["frozen_helpers"]) == set(protocol["runner_manifest"])
    for name, script in report["scripts"]["frozen_helpers"].items():
        assert (
            _sha(script["source"].encode()) == script["sha256"] == protocol["runner_manifest"][name]
        )
        compile(script["source"], name, "exec")
    for name in ("posthoc_saved_array_verifier", "paired_replay_driver"):
        script = report["scripts"][name]
        assert _sha(script["source"].encode()) == script["sha256"]
        compile(script["source"], name, "exec")
    bridge = source["final_source_bridge"]
    assert _canonical(bridge["files"]) == bridge["manifest_sha256"]
    changed = {name for name, digest in source["files"].items() if bridge["files"][name] != digest}
    assert (
        changed
        == set(bridge["changed_since_study"])
        == {
            "src/resdag/__init__.py",
            "src/resdag/core/workflow.py",
            "src/resdag/models/parallel_esn.py",
        }
    )
    for name, entry in bridge["changed_since_study"].items():
        assert entry["study_sha256"] == source["files"][name]
        assert entry["later_sha256"] == bridge["files"][name]
        assert entry["diff"]
        assert _sha(entry["later_source"].encode()) == entry["later_sha256"]


def test_independent_study_fresh_ics_and_complete_nested_design():
    report = _report()
    protocol = report["preregistration"]["protocol"]
    trajectories = protocol["trajectories"]
    assert len(trajectories) == len({t["id"] for t in trajectories}) == 10
    for system, seed in (("lorenz", 2026102306), ("rossler", 2026102307)):
        rng = random.Random(seed)
        selected = [t for t in trajectories if t["system"] == system]
        assert len(selected) == 5
        for index, trajectory in enumerate(selected):
            assert trajectory["ic_index"] == index
            assert trajectory["ic_generator_seed"] == seed
            assert trajectory["initial_condition"] == [rng.uniform(0.5, 1.5) for _ in range(3)]
    assert protocol["initialization_seeds"] == [5849201, 5849202, 5849203]
    assert len(protocol["recipes"]) == 6
    assert sum(r["panel"] == "primary-capability" for r in protocol["recipes"].values()) == 4
    assert (
        sum(r["panel"] == "secondary-common-state-only" for r in protocol["recipes"].values()) == 2
    )
    expected_forecasts, expected_fits = set(), set()
    for trajectory in trajectories:
        system = trajectory["system"]
        config = protocol["data_protocols"][system]
        assert config["training"][0] == 0
        assert config["washout"] == 500
        origins, horizon = config["test_origins"], config["horizon_steps"]
        assert origins == [config["test"][0] + k * horizon for k in range(3)]
        assert origins[-1] + horizon == config["test"][1]
        for recipe in (name for name in protocol["recipes"] if name.startswith(system + "-")):
            for seed in protocol["initialization_seeds"]:
                expected_fits.add((recipe, trajectory["id"], seed))
                expected_forecasts.update((recipe, trajectory["id"], seed, k) for k in range(3))
    rows, fits = report["forecasts"], report["fits"]
    assert len(rows) == len({r["id"] for r in rows}) == 270
    assert len(fits) == len({f["id"] for f in fits}) == 90
    assert {
        (r["recipe"], r["trajectory"], r["seed"], r["origin_id"]) for r in rows
    } == expected_forecasts
    assert {(f["recipe"], f["trajectory"], f["seed"]) for f in fits} == expected_fits
    assert all(r["fit_id"] in {f["id"] for f in fits} for r in rows)
    accounting = report["accounting"]
    assert accounting["planned_fits"] == accounting["completed_fits"] == 90
    assert accounting["planned_forecasts"] == accounting["retained_forecasts"] == 270
    assert accounting["excluded_forecasts"] == accounting["replacement_trajectories"] == []
    assert accounting["retuned_recipes"] == []
    assert accounting["fit_failures"] == sum(f["status"] != "complete" for f in fits) == 0
    assert (
        accounting["forecast_failures"]
        == sum(r["status"] not in ("complete", "diverged") for r in rows)
        == 0
    )
    assert accounting["diverged_forecasts"] == sum(r["status"] == "diverged" for r in rows) == 0
    data = report["data_manifest"]
    assert data["preregistered_sha256"] == report["preregistration"]["sha256"]
    assert data["all_data_sealed_before_fitting"]
    assert [r["trajectory"] for r in data["records"]] == trajectories
    assert len(report["initial_weight_manifest"]["records"]) == 18
    _assert_recipes_and_initial_weights_remain_frozen(report)


def _assert_recipes_and_initial_weights_remain_frozen(report):
    protocol = report["preregistration"]["protocol"]
    for relative, original in protocol["recipe_sources"].items():
        assert _sha((ROOT / relative).read_bytes()) == original["sha256"]
    capability = json.loads(
        (ROOT / "docs/assets/data/phase-five-capability-2026-10-23.json").read_text()
    )
    common = json.loads(
        (ROOT / "docs/assets/data/phase-five-equal-time-2026-10-23.json").read_text()
    )
    for name, recipe in protocol["recipes"].items():
        system = name.split("-")[0]
        expected = (
            capability["frozen_winners"][name]["winning_params"]
            if recipe["panel"] == "primary-capability"
            else common["tracks"][f"{system}-resdag-cpu"]["winner"]["params"]
        )
        assert recipe["params"] == expected
    weights = report["initial_weight_manifest"]["records"]
    for record in weights.values():
        fingerprint = record["initial_fingerprint"]
        assert _canonical(fingerprint["tensors"]) == fingerprint["sha256"]
    for fit in report["fits"]:
        reference = weights[f"{fit['recipe']}-seed{fit['seed']}"]
        assert fit["initial_fingerprint"] == reference["initial_fingerprint"]["sha256"]
    verification = report["independent_verification"]
    assert verification["additional_fits_or_forecasts"] == 0
    assert verification["fits"] == verification["fitted_checkpoint_files_verified"] == 90
    assert verification["forecasts"] == 270
    assert verification["initial_weight_files_verified"] == 18
    for key in (
        "all_planned_cells_present",
        "all_data_time_prediction_hashes_verified",
        "all540_threshold_first_crossings_verified",
        "all_trajectory_and_pooled_summaries_recomputed",
        "source_helper_manifests_unchanged",
    ):
        assert verification[key]


@pytest.mark.parametrize("threshold", ["0.3", "0.5"])
def test_independent_study_every_first_crossing_and_timeline(threshold):
    report = _report()
    protocol = report["preregistration"]["protocol"]
    for row in report["forecasts"]:
        config = protocol["data_protocols"][row["system"]]
        errors = _errors(row)
        assert all(math.isfinite(value) and value >= 0 for value in errors)
        prefix = next(
            (
                i
                for i, value in enumerate(errors)
                if not math.isfinite(value) or value >= float(threshold)
            ),
            len(errors),
        )
        score = row["scores"][threshold]
        assert score["steps"] == prefix
        assert score["right_censored"] is (prefix == len(errors))
        assert score["lyapunov_times"] == prefix * config["dt"] * config["lambda_max"]
        assert score["library_score_error"] is None
        rms = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
        assert score["full_horizon_nrmse"] == pytest.approx(rms, rel=2e-14)
        assert len(row["training_scale"]) == 3
        assert all(math.isfinite(scale) and scale > 0 for scale in row["training_scale"])
        assert row["origin"] == config["test_origins"][row["origin_id"]]
        assert row["horizon"] == config["horizon_steps"]
        assert row["prediction"]["shape"] == [len(errors), 3]
        assert row["history_steps"] == (
            500 if row["panel"] == "secondary-common-state-only" else row["origin"]
        )
        timeline = row["normalized_error_trace"]["timeline"]
        assert timeline["first_truth_index"] == row["origin"]
        assert timeline["last_truth_index"] == row["origin"] + len(errors) - 1
        assert timeline["dt"] == config["dt"]
        assert timeline["first_truth_time"] == pytest.approx(
            (row["origin"] + config["discard_samples"]) * config["dt"]
        )
        assert timeline["last_truth_time"] == pytest.approx(
            timeline["first_truth_time"] + (len(errors) - 1) * config["dt"]
        )


def test_independent_study_all_summaries_use_five_trajectory_units():
    report = _report()
    assert len(report["summaries"]) == 6
    for summary in report["summaries"]:
        rows = [row for row in report["forecasts"] if row["recipe"] == summary["recipe"]]
        assert len(rows) == summary["count"] == 45
        assert summary["failed_forecasts"] == sum(
            r["status"] in ("fit_failed", "forecast_failed") for r in rows
        )
        assert summary["diverged_forecasts"] == sum(r["status"] == "diverged" for r in rows)
        for threshold in ("0.3", "0.5"):
            recorded = summary["scores"][threshold]
            values = [r["scores"][threshold]["lyapunov_times"] for r in rows]
            trajectory_medians = {
                trajectory: statistics.median(
                    r["scores"][threshold]["lyapunov_times"]
                    for r in rows
                    if r["trajectory"] == trajectory
                )
                for trajectory in sorted({r["trajectory"] for r in rows})
            }
            assert len(trajectory_medians) == 5
            assert recorded["trajectory_medians"] == trajectory_medians
            assert recorded["primary_median_of_trajectory_medians"] == statistics.median(
                trajectory_medians.values()
            )
            assert recorded["trajectory_median_min"] == min(trajectory_medians.values())
            assert recorded["trajectory_median_max"] == max(trajectory_medians.values())
            assert recorded["pooled_median"] == statistics.median(values)
            assert recorded["individual_min"] == min(values)
            assert recorded["individual_max"] == max(values)
            assert recorded["right_censored"] == sum(
                r["scores"][threshold]["right_censored"] for r in rows
            )
            assert recorded["per_seed_medians"] == {
                str(seed): statistics.median(
                    r["scores"][threshold]["lyapunov_times"] for r in rows if r["seed"] == seed
                )
                for seed in sorted({r["seed"] for r in rows})
            }


def test_independent_study_exact_paired_regression_and_native_receipts():
    report = _report()
    paired = report["paired_regression"]
    assert paired["baseline_commit"] == "eb70bf11daf9d9d644bf53c26699042f169a223b"
    assert (
        _canonical(paired["baseline_source_manifest"]) == paired["baseline_source_manifest_sha256"]
    )
    assert paired["candidate_source_manifest_sha256"] == report["source"]["manifest_sha256"]
    assert paired["original_study_preregistered_sha256"] == report["preregistration"]["sha256"]
    forecasts = {r["id"]: r for r in report["forecasts"]}
    assert len(paired["forecast_comparisons"]) == 270
    assert {r["id"] for r in paired["forecast_comparisons"]} == set(forecasts)
    for comparison in paired["forecast_comparisons"]:
        candidate, baseline = forecasts[comparison["id"]], comparison["baseline"]
        for key in ("status", "prediction", "prediction_file", "final_state", "scores"):
            assert candidate[key] == baseline[key]
        assert all(
            comparison[key]
            for key in (
                "same_status",
                "same_complete_prediction_hash",
                "same_final_state_hashes",
                "same_scores",
                "same_full_prediction_file_bytes",
            )
        )
    fits = {f["id"]: f for f in report["fits"]}
    assert len(paired["fit_comparisons"]) == 90
    assert {f["id"] for f in paired["fit_comparisons"]} == set(fits)
    for comparison in paired["fit_comparisons"]:
        candidate, baseline = fits[comparison["id"]], comparison["baseline"]
        assert candidate["status"] == baseline["status"]
        assert candidate["initial_fingerprint"] == baseline["initial_fingerprint"]
        assert (
            _canonical(candidate["fitted_fingerprint"]["tensors"])
            == candidate["fitted_fingerprint"]["sha256"]
            == baseline["fitted_tensor_fingerprint_sha256"]
        )
        assert all(
            comparison[key]
            for key in ("same_status", "same_initial_fingerprint", "same_fitted_tensor_fingerprint")
        )
    assert paired["all270_full_forecasts_bit_identical"]
    assert paired["all90_fitted_tensor_fingerprints_identical"]
    _assert_final_runtime_replay(report, fits, forecasts)
    for process in (
        report["processes"]
        + paired["baseline_processes"]
        + report["final_runtime_replay"]["processes"]
    ):
        policy = process["native_policy"]
        assert policy["fresh_cli_bootstrap"]
        assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
        assert set(policy["requested_environment"].values()) == {"1"}
        assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
        assert process.get("timing_boundary", process.get("timing_scope"))
        assert "warnings" in process
        assert process["start_condition"]["utc"] <= process["end_condition"]["utc"]


def _assert_final_runtime_replay(report, fits, forecasts):
    replay = report["final_runtime_replay"]
    plan = replay["plan"]
    assert plan["source_files"] == report["source"]["final_source_bridge"]["files"]
    assert _canonical(plan["source_files"]) == plan["source_manifest_sha256"]
    assert plan["original_preregistered_sha256"] == report["preregistration"]["sha256"]
    assert plan["original_source_manifest_sha256"] == report["source"]["manifest_sha256"]
    assert replay["additional_fits"] == replay["additional_integrations"] == 0
    assert plan["additional_fits"] == plan["additional_ode_integrations"] == 0
    assert plan["selection_or_retuning"] is False
    assert len(replay["fitted_checkpoints"]) == 90
    assert {f["id"] for f in replay["fitted_checkpoints"]} == set(fits)
    for loaded in replay["fitted_checkpoints"]:
        original = fits[loaded["id"]]
        assert loaded["checkpoint_sha256"] == original["checkpoint"]["sha256"]
        expected = {
            key: value
            for key, value in original["fitted_fingerprint"]["tensors"].items()
            if key in loaded["persisted_tensor_names"]
        }
        assert _canonical(expected) == loaded["persisted_fitted_tensors_sha256"]
        assert loaded["all_persisted_fitted_tensors_exact"]
        assert (
            loaded["expected_fitted_tensor_fingerprint_sha256"]
            == original["fitted_fingerprint"]["sha256"]
        )
        if loaded["exact"]:
            assert (
                loaded["fitted_tensor_fingerprint_sha256"]
                == original["fitted_fingerprint"]["sha256"]
            )
            assert loaded["post_fit_transient_state_status"] == "exact"
        else:
            assert loaded["post_fit_transient_state_reconstruction_required"]
            assert loaded["post_fit_transient_state_status"].startswith("unavailable;")
    assert replay["all90_fitted_tensor_fingerprints_identical"] == all(
        f["exact"] for f in replay["fitted_checkpoints"]
    )
    assert replay["all90_persisted_fitted_tensor_fingerprints_identical"]
    assert replay["all90_fitted_tensor_fingerprints_identical"]
    assert len(replay["forecasts"]) == 270
    assert {r["id"] for r in replay["forecasts"]} == set(forecasts)
    for repeated in replay["forecasts"]:
        original = forecasts[repeated["id"]]
        for key in ("prediction", "final_state", "scores"):
            assert repeated[key] == original[key]
        assert repeated["prediction_npy_sha256"] == original["prediction_file"]["sha256"]
        assert repeated["error_trace_sha256"] == original["normalized_error_trace"]["sha256"]
        assert all(
            repeated[key]
            for key in (
                "same_prediction",
                "same_prediction_file_bytes",
                "same_final_states",
                "same_scores",
            )
        )
    assert replay["all270_full_forecasts_states_scores_bit_identical"]
    script = report["scripts"]["final_runtime_replay"]
    assert _sha(script["source"].encode()) == script["sha256"] == plan["replay_script_sha256"]
    compile(script["source"], "<final runtime replay>", "exec")
    assert len(replay["processes"]) == 10
    assert [attempt["retained_forecast_checks"] for attempt in report["replay_setup_attempts"]] == [
        18,
        18,
    ]
    for attempt in report["replay_setup_attempts"]:
        assert attempt["additional_fits"] == attempt["additional_integrations"] == 0
        assert attempt["all_retained_forecast_checks_exact"]
        assert _sha(attempt["script"]["source"].encode()) == attempt["script"]["sha256"]


def test_independent_study_reconstructs_frozen_tree_with_stdlib_only(tmp_path):
    report = _report()
    staged = report["reproduction"]["stage_script"]
    assert _sha(staged["source"].encode()) == staged["sha256"]
    tree = ast.parse(staged["source"])
    for node in ast.walk(tree):
        names = (
            [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.Import)
            else ([node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) else [])
        )
        assert all(name in sys.stdlib_module_names for name in names)
    script = tmp_path / "reconstruct_independent.py"
    script.write_text(staged["source"])
    output = tmp_path / "candidate"
    command = [
        sys.executable,
        "-S",
        str(script),
        "--publication",
        str(PUBLICATION),
        "--repository",
        str(ROOT),
        "--output",
        str(output),
        "--numerical-python",
        sys.executable,
    ]
    subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    receipt = json.loads((output / "staging-receipt.json").read_text())
    assert receipt["source_manifest_sha256"] == report["source"]["manifest_sha256"]
    assert (
        receipt["runner_manifest_sha256"]
        == report["preregistration"]["protocol"]["runner_manifest_sha256"]
    )
    assert receipt["preregistered_sha256"] == report["preregistration"]["sha256"]
    assert receipt["numerical_modules_imported"] == receipt["numerical_commands_executed"] == []
    assert _sha((output / "preregistered.json").read_bytes()) == report["preregistration"]["sha256"]
    assert list((output / "data").iterdir()) == list((output / "weights").iterdir()) == []
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert _sha((output / "preregistered.json").read_bytes()) == report["preregistration"]["sha256"]
    final_stage = report["reproduction"]["final_replay_stage_script"]
    assert _sha(final_stage["source"].encode()) == final_stage["sha256"]
    final_script = tmp_path / "reconstruct_final_replay.py"
    final_script.write_text(final_stage["source"])
    final_output = tmp_path / "final"
    final_command = [
        sys.executable,
        "-S",
        str(final_script),
        "--publication",
        str(PUBLICATION),
        "--repository",
        str(ROOT),
        "--output",
        str(final_output),
        "--numerical-python",
        sys.executable,
    ]
    subprocess.run(final_command, capture_output=True, text=True, check=True, timeout=30)
    final_receipt = json.loads((final_output / "staging-receipt.json").read_text())
    assert final_receipt["source_only"]
    assert (
        final_receipt["source_manifest_sha256"]
        == report["source"]["final_source_bridge"]["manifest_sha256"]
    )
    assert (
        final_receipt["replay_script_sha256"] == report["scripts"]["final_runtime_replay"]["sha256"]
    )
    assert (
        final_receipt["numerical_modules_imported"]
        == final_receipt["numerical_commands_executed"]
        == []
    )
    assert final_receipt["commands"] == []
    final_manifest = {
        path.relative_to(final_output / "source").as_posix(): _sha(path.read_bytes())
        for path in (final_output / "source").rglob("*")
        if path.is_file()
    }
    assert final_manifest == report["source"]["final_source_bridge"]["files"]
