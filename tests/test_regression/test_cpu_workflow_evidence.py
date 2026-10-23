"""Check published CPU evidence independently of the measurement summaries."""

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "docs/assets/data"


@pytest.mark.parametrize("profile", ["workflows", "storage"])
def test_published_cpu_process_samples_and_numerical_agreement(profile):
    report = json.loads((DATA / f"phase-five-cpu-{profile}-2026-10-23.json").read_text())
    protocol, processes = report["protocol"], report["processes"]
    assert protocol["sources"]["before"]["head_commit"].startswith("18f78c83")
    assert protocol["sources"]["after"]["head_commit"].startswith("dc695a6")
    assert (protocol["blocks"], protocol["repeats"], protocol["warmups"]) == (2, 7, 1)
    assert len(processes) == 8
    assert [process["label"] for process in processes] == ["before", "after", "after", "before"] * 2
    assert report["all_hashes_equal"]
    expected_rows = 7 if profile == "workflows" else 6
    assert len(report["comparisons"]) == expected_rows
    for process in processes:
        policy = process["native_thread_policy"]
        assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
        assert policy["fresh_cli_bootstrap"]
        assert set(policy["requested_environment"].values()) == {"1"}
        assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
        assert len(process["rows"]) == expected_rows
        for row in process["rows"]:
            assert len(row["runs"]) == 8
            assert sum(run["warmup"] for run in row["runs"]) == 1
            timed = [run for run in row["runs"] if not run["warmup"]]
            for metric, summary in row["summary"].items():
                values = [run["timings"][metric] for run in timed]
                assert all(math.isfinite(value) and value > 0 for value in values)
                assert summary["median"] == statistics.median(values)
    for case, comparison in report["comparisons"].items():
        rows = [
            next(row for row in process["rows"] if row["case"]["id"] == case)
            for process in processes
        ]
        checks = [run["hashes"] for row in rows for run in row["runs"]]
        assert all(check == checks[0] for check in checks)
        assert comparison["all_hashes_equal"]
        for metric, result in comparison["metrics"].items():
            medians = [row["summary"][metric]["median"] for row in rows]
            ratios = [medians[a] / medians[b] for a, b in ((0, 1), (3, 2), (4, 5), (7, 6))]
            assert result["paired_process_ratios_before_over_after"] == ratios
            assert result["ratio_summary"]["median"] == statistics.median(ratios)
    if profile == "storage":
        assert len(report["storage_diagnostics"]) == 24
        for diagnostic in report["storage_diagnostics"]:
            assert diagnostic["initial_logical_weights_equal"]
            assert math.isfinite(diagnostic["complete_forecast_max_abs_difference"])


def test_published_cpu_forecast_replay_is_complete_and_exact():
    report = json.loads((DATA / "phase-five-cpu-accuracy-2026-10-23.json").read_text())
    assert set(report["systems"]) == {"lorenz", "rossler"}
    assert report["source"] == "final-runtime-manifest"
    assert len(report["runtime_manifest"]) == 205
    manifest = json.dumps(report["runtime_manifest"], indent=2) + "\n"
    assert hashlib.sha256(manifest.encode()).hexdigest() == report["runtime_manifest_sha256"]
    for system, result in report["systems"].items():
        assert len(result["forecasts"]) == 15
        assert result["all_equal"]
        for forecast in result["forecasts"].values():
            assert forecast["equal"]
            assert forecast["max_abs_difference"] == 0
            assert forecast["shape"] == [553 if system == "lorenz" else 966, 3]


def test_published_readout_probe_preserves_coefficients_and_diagnostics():
    report = json.loads((DATA / "phase-five-cpu-auto-2026-10-23.json").read_text())
    processes = report["processes"]
    assert [process["label"] for process in processes] == ["before", "after", "after", "before"] * 2
    for index, summary in enumerate(report["summaries"]):
        medians, checks = [], []
        for process in processes:
            row = process["rows"][index]
            assert (row["n"], row["features"]) == (summary["n"], summary["features"])
            assert len(row["runs"]) == 8
            assert sum(run["warmup"] for run in row["runs"]) == 1
            medians.append(
                statistics.median(run["seconds"] for run in row["runs"] if not run["warmup"])
            )
            checks.extend(
                (run["weight_sha256"], run["bias_sha256"], run["report"]) for run in row["runs"]
            )
        assert all(check == checks[0] for check in checks)
        assert summary["coefficients_intercepts_reports_exact"]
        ratios = [medians[a] / medians[b] for a, b in ((0, 1), (3, 2), (4, 5), (7, 6))]
        assert summary["process_medians_seconds"] == medians
        assert summary["paired_ratios_before_over_after"] == ratios
        assert summary["median_ratio"] == statistics.median(ratios)


def test_published_compile_probe_separates_first_request_from_warm_requests():
    report = json.loads((DATA / "phase-five-cpu-compile-2026-10-23.json").read_text())
    assert [row["horizon"] for row in report["rows"]] == [257, 1025]
    for row in report["rows"]:
        assert row["compiled_cache_entries"] == 1
        assert row["eager_sha256"] == row["compiled_sha256"]
        assert row["max_abs_difference"] == 0
        assert len(row["eager_seconds"]) == len(row["warm_compiled_requests_seconds"]) == 7
        assert row["eager_median"] == statistics.median(row["eager_seconds"])
        assert row["warm_median"] == statistics.median(row["warm_compiled_requests_seconds"])
        assert math.isfinite(row["first_compiled_request_seconds"])
        assert row["first_compiled_request_seconds"] > 0


def _recurrence_report():
    return json.loads((DATA / "phase-five-cpu-recurrence-2026-10-23.json").read_text())


def test_recurrence_evidence_binds_frozen_source_scripts_and_environment():
    report = _recurrence_report()
    source = report["source"]
    assert source["commit"] == "3b34005efd93a2f6573ecd96c9a5b23daecdf547"
    assert len(source["files"]) == 205
    assert hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest() == (
        source["manifest_sha256"]
    )
    assert source["runtime_tree_identical"] and source["files_unchanged_after_measurement"]
    for script in report["scripts"].values():
        assert hashlib.sha256(script["source"].encode()).hexdigest() == script["sha256"]
        compile(script["source"], "<recorded diagnostic>", "exec")
    for key, policy in report["native_policies"].items():
        assert hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest() == key
        assert policy["fresh_cli_bootstrap"]
        assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
        assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
        assert set(policy["requested_environment"].values()) == {"1"}
    for key, origins in report["import_origin_manifests"].items():
        assert hashlib.sha256(json.dumps(origins, sort_keys=True).encode()).hexdigest() == key


def test_recurrence_attribution_keeps_public_timings_separate_from_profiles():
    report = _recurrence_report()
    assert len(report["initial_attribution"]) == 4
    for case in report["initial_attribution"]:
        assert case["source_manifest_sha256"] == report["source"]["manifest_sha256"]
        assert case["probe_sha256"] == report["scripts"]["probe.py"]["sha256"]
        assert case["native_policy_sha256"] in report["native_policies"]
        assert case["import_origins_sha256"] in report["import_origin_manifests"]
        runs = case["public_runs"]
        assert len(runs) == 6 and sum(run["warmup"] for run in runs) == 1
        for operation, summary in case["public_summary"].items():
            values = [run[operation]["wall_seconds"] for run in runs if not run["warmup"]]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert summary["median"] == statistics.median(values)
            assert (summary["min"], summary["max"]) == (min(values), max(values))


@pytest.mark.parametrize("name,rounds", [("b4", 7), ("csr-bridge", 4)])
def test_recurrence_candidate_tables_recompute_from_every_abba_sample(name, rounds):
    report = _recurrence_report()
    processes = report["whole_workflow_diagnostics"][name]["processes"]
    assert len(processes) == 2
    assert len({process["pid"] for process in processes}) == 2
    hashes_by_mode = {}
    for process in processes:
        assert process["source_manifest_sha256"] == report["source"]["manifest_sha256"]
        script = "b4_workflow.py" if name == "b4" else "csr_workflow.py"
        assert process["probe_sha256"] == report["scripts"][script]["sha256"]
        assert process["native_policy_sha256"] in report["native_policies"]
        rows = process["rows"]
        assert len(rows) == 4 * rounds
        for index, row in enumerate(rows):
            assert (row["round"], row["slot"]) == divmod(index, 4)
            assert row["mode"] == ("before", "after", "after", "before")[index % 4]
            for metric in ("fit_wall", "fit_cpu", "forecast_wall", "forecast_cpu"):
                assert math.isfinite(row[metric]) and row[metric] > 0
            expected = hashes_by_mode.setdefault(row["mode"], row["hashes"])
            assert row["hashes"] == expected == process["same_mode_hashes"][row["mode"]]
        for metric, summary in process["summary"].items():
            before = [row[metric] for row in rows if row["mode"] == "before"]
            after = [row[metric] for row in rows if row["mode"] == "after"]
            ratios = []
            for start in range(0, len(rows), 4):
                ratios.extend(
                    rows[start + left][metric] / rows[start + right][metric]
                    for left, right in ((0, 1), (3, 2))
                )
            assert summary["before_median"] == statistics.median(before)
            assert summary["after_median"] == statistics.median(after)
            assert summary["ratios"] == ratios
            assert summary["paired_before_after_median"] == statistics.median(ratios)
            assert (summary["paired_min"], summary["paired_max"]) == (min(ratios), max(ratios))
        assert process["cross_mode_exact"] is (name == "csr-bridge")
        difference = process["cross_mode_maxabs"]
        if name == "csr-bridge":
            assert hashes_by_mode["before"] == hashes_by_mode["after"]
            assert difference["prediction"] == difference["coefficients"] == 0
            assert all(value == 0 for value in difference["states"].values())
        else:
            assert hashes_by_mode["before"] != hashes_by_mode["after"]
            assert difference["prediction"] > 4000 and difference["coefficients"] > 10


def test_recurrence_documentation_table_matches_recorded_processes():
    page = (DATA.parents[1] / "project/phase-five-results.md").read_text()
    processes = _recurrence_report()["whole_workflow_diagnostics"]["csr-bridge"]["processes"]
    for index, process in enumerate(processes, start=1):
        fit, forecast = (process["summary"][name] for name in ("fit_wall", "forecast_wall"))
        expected = (
            f"| {index} | {fit['before_median'] * 1000:.2f} → "
            f"{fit['after_median'] * 1000:.2f} ms | "
            f"{fit['paired_before_after_median']:.3f} "
            f"[{fit['paired_min']:.3f}–{fit['paired_max']:.3f}] | "
            f"{forecast['before_median'] * 1000:.2f} → "
            f"{forecast['after_median'] * 1000:.2f} ms | "
            f"{forecast['paired_before_after_median']:.3f} "
            f"[{forecast['paired_min']:.3f}–{forecast['paired_max']:.3f}] |"
        )
        assert expected in page


def test_recurrence_diagnostic_reconstructs_exact_inputs_without_numerical_imports(tmp_path):
    report = _recurrence_report()
    staged = report["reproduction"]["stage_script"]
    assert hashlib.sha256(staged["source"].encode()).hexdigest() == staged["sha256"]
    script = tmp_path / "reconstruct_cpu_profile.py"
    script.write_text(staged["source"])
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-S",
        str(script),
        "--publication",
        str(DATA / "phase-five-cpu-recurrence-2026-10-23.json"),
        "--repository",
        str(DATA.parents[2]),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    receipt = json.loads(completed.stdout)
    assert receipt["source_files"] == 205
    assert receipt["source_manifest_sha256"] == report["source"]["manifest_sha256"]
    assert receipt["script_sha256"] == {
        name: entry["sha256"] for name, entry in report["scripts"].items()
    }
    assert receipt["numerical_modules_imported"] == []
    assert receipt["numerical_commands_executed"] is False
    protocol_bytes = (output / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["source_files"] == report["source"]["files"]
    assert protocol["manifest_sha256"] == report["source"]["manifest_sha256"]
    assert protocol["cases"] == report["protocol"]["cases"]
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert (output / "protocol.json").read_bytes() == protocol_bytes
