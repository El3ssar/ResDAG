"""Non-numerical protocol oracles for the fixed-artifact CPU measurement tool."""

import json
import math
import os
import subprocess
import sys

import pytest

from benchmarks import cpu_workflows as cpu


def test_full_matrix_keeps_batch_scaling_workload_explicit(tmp_path):
    args = cpu._arguments(["--before", "HEAD", "--artifacts-dir", str(tmp_path / "new")])
    assert (args.blocks, args.repeats, args.warmups) == (2, 7, 1)
    cases = cpu._cases(args)
    assert len(cases) == 7
    assert {(row["input_steps"], row["horizon"]) for row in cases[:4]} == {(1501, 257)}
    assert {(row["input_steps"], row["horizon"]) for row in cases[4:]} == {(501, 129)}
    assert [row["batch"] for row in cases[4:]] == [1, 4, 16]
    assert len({row["artifact_key"] for row in cases[4:]}) == 1


def test_smoke_is_a_distinct_explicit_matrix_with_override_support(tmp_path):
    args = cpu._arguments(
        [
            "--before",
            "HEAD",
            "--artifacts-dir",
            str(tmp_path / "new"),
            "--profile",
            "smoke",
            "--repeats",
            "3",
        ]
    )
    assert (args.blocks, args.repeats, args.warmups) == (1, 3, 1)
    assert {row["reservoir_size"] for row in cpu._cases(args)} == {32, 64}
    assert {row["horizon"] for row in cpu._cases(args)} == {17}


def test_storage_matrix_pairs_identical_baseline_artifacts_at_each_width(tmp_path):
    args = cpu._arguments(
        ["--before", "HEAD", "--artifacts-dir", str(tmp_path / "new"), "--profile", "storage"]
    )
    cases = cpu._cases(args)
    assert len(cases) == 6
    assert [row["reservoir_size"] for row in cases] == [400, 400, 1000, 1000, 2000, 2000]
    assert {row["batch"] for row in cases} == {1}
    assert {
        (row["input_steps"], row["washout"], row["history_steps"], row["horizon"]) for row in cases
    } == {(501, 100, 100, 129)}
    for dense, csr in zip(cases[::2], cases[1::2]):
        assert dense["storage"] == "dense" and not dense["sparse"]
        assert csr["storage"] == "csr" and csr["sparse"]
        assert dense["artifact_key"] == csr["artifact_key"]
        assert dense["storage_pair"] == csr["storage_pair"]


def test_selected_rows_must_belong_to_the_profile(tmp_path):
    with pytest.raises(SystemExit):
        cpu._arguments(
            [
                "--before",
                "HEAD",
                "--artifacts-dir",
                str(tmp_path / "new"),
                "--profile",
                "storage",
                "--only",
                "classic-400-b1",
            ]
        )


@pytest.mark.parametrize("case_name", ["classic-400-b1", "ott-400-b1", "cycle-400-b1", "er-400-b1"])
def test_measurement_placeholder_never_invokes_spectral_normalization(
    tmp_path, monkeypatch, case_name
):
    import torch

    import resdag as rd
    from resdag.init.topology import base as topology_base

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Measurement placeholders must load stored weights, not normalize discarded weights"
        )

    monkeypatch.setattr(topology_base, "scale_to_spectral_radius", forbidden)
    args = cpu._arguments(
        [
            "--before",
            "HEAD",
            "--artifacts-dir",
            str(tmp_path / "new"),
            "--profile",
            "smoke",
            "--only",
            case_name,
        ]
    )
    graph = cpu._build(cpu._cases(args)[0], rd, torch)
    reservoir = next(layer for layer in graph.modules() if isinstance(layer, rd.BaseReservoirLayer))
    assert reservoir.cell.spectral_radius is None


def test_initial_logical_recurrence_handles_csr_zero_signs_without_weakening_exact_hashes(tmp_path):
    import torch

    import resdag as rd

    args = cpu._arguments(
        [
            "--before",
            "HEAD",
            "--artifacts-dir",
            str(tmp_path / "new"),
            "--profile",
            "smoke",
            "--only",
            "er-400-b1",
        ]
    )
    case = cpu._cases(args)[0]
    dense = cpu._build(case, rd, torch, reference=True)
    sparse = cpu._build(case, rd, torch)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    raw_dense = cpu._model_fingerprints(dense, rd)
    raw_sparse = cpu._model_fingerprints(sparse, rd)
    assert raw_dense["sha256"] != raw_sparse["sha256"]
    assert cpu._model_fingerprints(dense, rd, logical_recurrence=True) == cpu._model_fingerprints(
        sparse, rd, logical_recurrence=True
    )
    # The general prediction/state fingerprint keeps signed zero significant.
    negative_zero = torch.tensor([-0.0], dtype=torch.float64)
    positive_zero = torch.tensor([0.0], dtype=torch.float64)
    assert cpu._fingerprints({"prediction": negative_zero}) != cpu._fingerprints(
        {"prediction": positive_zero}
    )
    assert cpu._fingerprints({"state": negative_zero}) != cpu._fingerprints(
        {"state": positive_zero}
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--blocks", "0"],
        ["--repeats", "-1"],
        ["--timeout", "nan"],
        ["--washout", "501"],
        ["--history-steps", "1000"],
        ["--only", "classic-400-b1", "classic-400-b1"],
    ],
)
def test_invalid_protocol_fails_before_creating_outputs(tmp_path, extra):
    destination = tmp_path / "new"
    with pytest.raises(SystemExit):
        cpu._arguments(["--before", "HEAD", "--artifacts-dir", str(destination), *extra])
    assert not destination.exists()


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [math.nan], [math.inf]])
def test_summary_rejects_invalid_timings(values):
    with pytest.raises(ValueError):
        cpu._summary(values)


def _processes():
    processes = []
    for block, timings in enumerate(((2, 1, 2, 8), (6, 1, 1, 8))):
        for slot, label in enumerate(("before", "after", "after", "before")):
            processes.append(
                {
                    "run_id": f"block{block}-slot{slot}",
                    "label": label,
                    "rows": [
                        {
                            "case": {"id": "cell"},
                            "runs": [
                                {
                                    "hashes": dict.fromkeys(
                                        (
                                            "fitted_model",
                                            "fit_state",
                                            "prediction",
                                            "forecast_state",
                                        ),
                                        "same",
                                    )
                                }
                            ],
                            "summary": {"fit_seconds": {"median": timings[slot]}},
                        }
                    ],
                }
            )
    return processes


def test_summary_pairs_ab_and_ba_process_medians_and_retains_block_spread():
    result = cpu._compare(_processes(), [{"id": "cell"}], 2)["cell"]
    metric = result["metrics"]["fit_seconds"]
    assert metric["paired_process_ratios_before_over_after"] == [2, 4, 6, 8]
    assert metric["ratio_summary"]["median"] == 5
    assert metric["block_median_ratios"] == [3, 7]
    assert metric["block_spread"]["min"] == 3
    assert metric["process_medians_seconds"]["before"] == [2, 8, 6, 8]
    assert result["all_hashes_equal"]


def test_numerical_mismatch_is_reported_even_for_a_discarded_run():
    processes = _processes()
    check = dict(processes[-1]["rows"][0]["runs"][0]["hashes"])
    check["prediction"] = "different"
    processes[-1]["rows"][0]["runs"].append({"warmup": True, "hashes": check})
    assert not cpu._compare(processes, [{"id": "cell"}], 2)["cell"]["all_hashes_equal"]


def test_missing_duplicate_and_misordered_processes_are_rejected():
    processes = _processes()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        cpu._compare(processes[:-1], [{"id": "cell"}], 2)
    processes[0]["label"] = "after"
    with pytest.raises(ValueError, match="revision order"):
        cpu._compare(processes, [{"id": "cell"}], 2)


def test_source_parity_does_not_require_cross_storage_arithmetic_identity():
    processes = _processes()
    for process in processes:
        original = process["rows"].pop()
        for storage in ("dense", "csr"):
            process["rows"].append(
                {
                    **original,
                    "case": {"id": storage},
                    "runs": [
                        {
                            "hashes": dict.fromkeys(
                                ("fitted_model", "fit_state", "prediction", "forecast_state"),
                                storage,
                            )
                        }
                    ],
                }
            )
    comparisons = cpu._compare(processes, [{"id": "dense"}, {"id": "csr"}], 2)
    assert all(row["all_hashes_equal"] for row in comparisons.values())


def test_existing_artifact_directory_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        cpu.main(["--before", "HEAD", "--artifacts-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_reserved_output_collision_is_rejected_before_preparation(tmp_path):
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="top-level report"):
        cpu.main(
            [
                "--before",
                "HEAD",
                "--artifacts-dir",
                str(destination),
                "--output",
                str(destination / "protocol.json"),
            ]
        )
    assert not destination.exists()


def test_real_cli_smoke_uses_fixed_weights_public_workflow_and_fresh_native_policy(tmp_path):
    pytest.importorskip("threadpoolctl")
    destination = tmp_path / "paired"
    environment = os.environ | {
        name: "8"
        for name in (
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "JULIA_NUM_THREADS",
        )
    }
    environment.pop("RESDAG_BENCHMARK_THREAD_BOOTSTRAPPED", None)
    result = subprocess.run(
        [
            sys.executable,
            str(cpu.ROOT / "benchmarks/cpu_workflows.py"),
            "--before",
            "HEAD",
            "--after",
            "WORKTREE",
            "--artifacts-dir",
            str(destination),
            "--profile",
            "smoke",
            "--only",
            "cycle-400-b1",
            "er-400-b1",
            "classic-1000-b4",
            "--blocks",
            "1",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--timeout",
            "120",
        ],
        cwd=cpu.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    failures = "\n".join(
        path.read_text() for path in sorted((destination / "processes").glob("*.stderr.txt"))
    )
    assert result.returncode == 0, result.stdout + result.stderr + failures
    report = json.loads((destination / "results.json").read_text())
    assert report["all_hashes_equal"]
    assert len(report["processes"]) == 4
    assert [process["label"] for process in report["processes"]] == [
        "before",
        "after",
        "after",
        "before",
    ]
    expected_weights = report["preparation"]["rows"]["weights"]
    for process in report["processes"]:
        policy = process["native_thread_policy"]
        assert policy["fresh_cli_bootstrap"]
        assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
        assert all(value == "1" for value in policy["requested_environment"].values())
        assert process["loaded_resdag_modules"]
        assert len(process["rows"]) == 3
        for row in process["rows"]:
            assert (
                row["initial_model"]
                == expected_weights[row["case"]["artifact_key"]]["logical_model"]
            )
            assert len(row["runs"]) == 1
            assert row["runs"][0]["timings"]["fit_seconds"] > 0
            assert row["runs"][0]["timings"]["history_forecast_seconds"] > 0
            assert set(row["runs"][0]["hashes"]) == {
                "fitted_model",
                "fit_state",
                "prediction",
                "forecast_state",
            }
