"""Public parallel CPU evidence is internally complete and independently readable."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
from pathlib import Path

import pytest

REPORT = (
    Path(__file__).resolve().parents[2] / "docs/assets/data/phase-five-parallel-cpu-2026-10-23.json"
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text())


def _assert_digest(value):
    assert re.fullmatch("[0-9a-f]{64}", value)


def test_scripts_and_source_manifests_reconstruct_recorded_variants(report):
    protocol = report["protocol"]
    assert report["schema"] == 1
    assert report["kind"] == "parallel-cpu-forecast-evidence"
    assert (
        _sha((json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode())
        == report["protocol_sha256"]
    )
    for name, script in report["scripts"].items():
        assert _sha(script["source"].encode()) == script["sha256"]
        compile(script["source"], name, "exec")
    assert report["scripts"]["probe.py"]["sha256"] == protocol["frozen_probe_sha256"]
    assert report["scripts"]["final_compare.py"]["sha256"] == protocol["driver_sha256"]
    for variant, manifest in protocol["source_files"].items():
        reconstructed = dict(report["base_source_files"])
        for path, override in report["source_overrides"][variant].items():
            assert not Path(path).is_absolute() and ".." not in Path(path).parts
            assert _sha(override["source"].encode()) == override["sha256"]
            compile(override["source"], path, "exec")
            reconstructed[path] = override["sha256"]
        assert reconstructed == manifest
        assert (
            _sha(json.dumps(manifest, sort_keys=True).encode())
            == protocol["source_manifest_sha256"][variant]
        )
        for digest in manifest.values():
            _assert_digest(digest)
    differences = [
        name
        for name, digest in protocol["source_files"]["baseline"].items()
        if protocol["source_files"]["candidate"][name] != digest
    ]
    assert differences == protocol["source_differences"] == ["src/resdag/models/parallel_esn.py"]


def test_full_abba_matrix_and_every_raw_sample_are_accounted_for(report):
    cases = {case["id"]: case for case in report["protocol"]["cases"]}
    assert {(case["batch"], case["patches"]) for case in cases.values()} == {
        (1, 8),
        (1, 64),
        (4, 8),
        (4, 64),
    }
    for case in cases.values():
        assert (
            case["units"],
            case["steps"],
            case["washout"],
            case["history"],
            case["horizon"],
        ) == (150, 501, 100, 100, 129)
        assert case["features"] == case["patches"] * 4
    processes = {row["id"]: row for row in report["processes"]}
    assert len(processes) == 8
    assert {(row["round"], row["slot"]) for row in processes.values()} == {
        (round_index, slot) for round_index in range(2) for slot in range(4)
    }
    for row in processes.values():
        assert row["variant"] == ("baseline" if row["slot"] in (0, 3) else "candidate")
        assert (
            row["source_manifest_sha256"]
            == report["protocol"]["source_manifest_sha256"][row["variant"]]
        )
    samples = report["samples"]
    assert len(samples) == 96
    assert len({(row["process"], row["case"], row["iteration"]) for row in samples}) == 96
    for process_id, process in processes.items():
        selected = [row for row in samples if row["process"] == process_id]
        assert {(row["case"], row["iteration"]) for row in selected} == {
            (case, iteration) for case in cases for iteration in (0, 1, 2)
        }
        for row in selected:
            assert row["warmup"] is (row["iteration"] == 0)
            assert row["variant"] == process["variant"] and row["round"] == process["round"]
            for metric in ("fit_seconds", "history_forecast_seconds"):
                assert math.isfinite(row[metric]) and row[metric] > 0
    assert sum(not row["warmup"] for row in samples) == 64
    assert report["summary"]["counts"] == {
        "cases": 4,
        "fresh_processes": 8,
        "timed_fit_forecast_pairs": 64,
        "untimed_warmup_fit_forecast_pairs": 32,
    }


def test_every_sample_binds_complete_exactness_and_fixture_hashes(report):
    cases = {case["id"]: case for case in report["protocol"]["cases"]}
    assert set(report["exactness"]) == set(report["fixture_fingerprints"]) == set(cases)
    for case_id, row in report["exactness"].items():
        hashes = row["hashes"]
        assert set(hashes) == {"fitted", "fit_state", "prediction", "forecast_state"}
        assert _sha(_canonical(hashes)) == row["sha256"]
        assert hashes["prediction"]["shape"] == [
            cases[case_id]["batch"],
            129,
            cases[case_id]["features"],
        ]
        assert hashes["prediction"]["dtype"] == "torch.float64"
        _assert_digest(hashes["prediction"]["sha256"])
        groups = [hashes[key] for key in ("fitted", "fit_state", "forecast_state")]
        groups.append(report["fixture_fingerprints"][case_id]["initial"])
        assert hashes["fitted"]["tensors"] and hashes["forecast_state"]["tensors"]
        for group in groups:
            assert _sha(json.dumps(group["tensors"], sort_keys=True).encode()) == group["sha256"]
            for tensor in group["tensors"].values():
                _assert_digest(tensor["sha256"])
        data = report["fixture_fingerprints"][case_id]["data"]
        assert data["shape"] == [cases[case_id]["batch"], 501, cases[case_id]["features"]]
        assert data["dtype"] == "torch.float64"
        _assert_digest(data["sha256"])
    for sample in report["samples"]:
        assert sample["exactness_id"] == sample["case"]
        assert sample["exactness_sha256"] == report["exactness"][sample["case"]]["sha256"]
    assert report["summary"]["all_hashes_exact"] is True


def test_ratios_are_independently_recomputed_from_full_samples(report):
    comparisons = report["summary"]["comparisons"]
    assert len(comparisons) == 8
    for comparison in comparisons:
        selected = [
            row
            for row in report["samples"]
            if row["round"] == comparison["round"]
            and row["case"] == comparison["case"]
            and not row["warmup"]
        ]
        for metric in ("fit_seconds", "history_forecast_seconds"):
            values = {
                variant: [row[metric] for row in selected if row["variant"] == variant]
                for variant in ("baseline", "candidate")
            }
            assert len(values["baseline"]) == len(values["candidate"]) == 4
            before, after = (statistics.median(values[key]) for key in ("baseline", "candidate"))
            assert comparison[metric] == {
                "baseline_seconds": before,
                "candidate_seconds": after,
                "speedup": before / after,
            }
        assert comparison["all_hashes_exact"] is True


def test_each_process_has_verified_single_thread_native_policy(report):
    for process in report["processes"]:
        policy = report["native_policies"][process["native_policy_id"]]
        assert _sha(_canonical(policy)) == process["native_policy_id"]
        assert (
            policy["workers"]
            == policy["torch_intraop_threads"]
            == policy["torch_interop_threads"]
            == 1
        )
        assert policy["dtype"] == "float64" and policy["fresh_cli_bootstrap"] is True
        assert all(value == "1" for value in policy["requested_environment"].values())
        assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
        for config in policy["build_configurations"].values():
            for kind in ("blas", "lapack"):
                name = config["Build Dependencies"][kind]["name"].lower()
                if "accelerate" in name:
                    assert policy["accelerate"]["BLASGetThreading"] == 1
                    assert policy["accelerate"]["scope"] == "calling thread"
                else:
                    assert any(pool["internal_api"] in name for pool in policy["native_pools"])


def test_replay_extraction_and_relocation_need_no_private_directory(report, tmp_path, monkeypatch):
    namespace = {"__name__": "public_evidence_replay"}
    exec(
        compile(report["scripts"]["public_replay.py"]["source"], "public_replay.py", "exec"),
        namespace,
    )
    base = b"value = 1\n"
    replacement = "value = 2\n"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as handle:
        item = tarfile.TarInfo("src/example.py")
        item.size = len(base)
        handle.addfile(item, io.BytesIO(base))
    monkeypatch.setattr(
        namespace["subprocess"], "check_output", lambda *args, **kwargs: archive.getvalue()
    )
    fixture = copy.deepcopy(report)
    fixture["protocol"]["source_files"] = {
        "baseline": {"src/example.py": _sha(base)},
        "candidate": {"src/example.py": _sha(replacement.encode())},
    }
    fixture["source_overrides"] = {
        "baseline": {},
        "candidate": {
            "src/example.py": {"source": replacement, "sha256": _sha(replacement.encode())}
        },
    }
    destination = tmp_path / "replay"
    namespace["extract"](fixture, tmp_path, destination)
    assert (destination / "final/baseline-source/src/example.py").read_bytes() == base
    assert (destination / "final/candidate-source/src/example.py").read_text() == replacement
    previous = {name: sys.modules.get(name) for name in ("probe", "final_compare")}
    try:
        probe, driver = namespace["configure"](destination, tmp_path)
        assert probe.ROOT == driver.ROOT == destination
        assert probe.REPO == tmp_path
        assert driver.FINAL == destination / "final"
        assert probe.PYTHON == Path(sys.executable)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
