"""Check published CPU evidence independently of the measurement summaries."""

import hashlib
import json
import math
import statistics
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
