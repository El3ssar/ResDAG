"""Independent fresh-process and fail-closed native thread-policy oracles."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.rc_bench import _threads

ROOT = Path(__file__).resolve().parents[1]
NATIVE = importlib.util.find_spec("threadpoolctl") is not None


def test_bootstrap_overrides_every_inherited_vendor_limit_before_restart(monkeypatch):
    for name in _threads.THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    monkeypatch.delenv(_threads._BOOTSTRAPPED, raising=False)
    original = [sys.executable, "-m", "benchmarks.rc_bench.budget", "--help"]
    monkeypatch.setattr(sys, "orig_argv", original)
    captured = {}

    def execute(executable, argv, environment):
        captured.update(executable=executable, argv=argv, environment=environment)

    monkeypatch.setattr(os, "execvpe", execute)
    _threads.bootstrap_cli()
    assert captured["argv"] == original
    assert all(captured["environment"][name] == "1" for name in _threads.THREAD_VARIABLES)
    assert captured["environment"][_threads._BOOTSTRAPPED] == "1"


def test_verified_bootstrap_does_not_restart_recursively(monkeypatch):
    monkeypatch.setenv(_threads._BOOTSTRAPPED, "1")
    for name in _threads.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(os, "execvpe", lambda *args: pytest.fail("recursive restart"))
    _threads.bootstrap_cli()


@pytest.mark.parametrize("backend", ["unknown", "", "Accelerate", "openblas"])
def test_unknown_or_unverified_native_backend_is_rejected(backend):
    config = {
        "numpy": {"Build Dependencies": {name: {"name": backend} for name in ("blas", "lapack")}}
    }
    with pytest.raises(_threads.ThreadPolicyError):
        _threads._verify_backends(config, [], None)


@pytest.mark.parametrize("counter", [None, 0, 2, 8])
def test_detected_native_pool_must_report_exactly_one(counter):
    config = {
        "numpy": {
            "Build Dependencies": {name: {"name": "scipy-openblas"} for name in ("blas", "lapack")}
        }
    }
    pools = [{"user_api": "blas", "internal_api": "openblas", "num_threads": counter}]
    with pytest.raises(_threads.ThreadPolicyError, match="counter"):
        _threads._verify_backends(config, pools, None)


@pytest.mark.skipif(not NATIVE, reason="install optional benchmark requirements")
def test_real_cli_overrides_eight_before_numpy_import(tmp_path):
    output = tmp_path / "imports.jsonl"
    (tmp_path / "sitecustomize.py").write_text(
        "import os,sys,json\n"
        "class Probe:\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        "  if fullname=='numpy':\n"
        "   with open(os.environ['THREAD_IMPORT_PROBE'],'a') as out:\n"
        "    print(json.dumps({k:os.environ.get(k) for k in os.environ if k.endswith('_NUM_THREADS') or k=='VECLIB_MAXIMUM_THREADS'}),file=out)\n"
        "  return None\n"
        "sys.meta_path.insert(0,Probe())\n"
    )
    environment = os.environ | {name: "8" for name in _threads.THREAD_VARIABLES}
    environment.pop(_threads._BOOTSTRAPPED, None)
    environment.update(
        THREAD_IMPORT_PROBE=str(output),
        PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH','')}",
    )
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.rc_bench.budget", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    imports = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(imports) == 1
    assert all(imports[0][name] == "1" for name in _threads.THREAD_VARIABLES)


@pytest.mark.skipif(not NATIVE, reason="install optional benchmark requirements")
def test_fresh_native_policy_records_and_retains_caps_after_numpy_scipy_torch_work():
    source = """
import json
from benchmarks.rc_bench._threads import bootstrap_cli, single_thread_policy, verify_thread_policy
bootstrap_cli()
with single_thread_policy() as policy:
 import numpy as np, scipy.linalg, torch
 scipy.linalg.solve(np.eye(3),np.ones(3))
 np.ones((3,3))@np.ones((3,3))
 torch.ones(3,3)@torch.ones(3,3)
 verify_thread_policy()
 print(json.dumps(policy))
"""
    environment = os.environ | {name: "8" for name in _threads.THREAD_VARIABLES}
    environment.pop(_threads._BOOTSTRAPPED, None)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    policy = json.loads(result.stdout)
    assert policy["fresh_cli_bootstrap"]
    assert policy["torch_intraop_threads"] == policy["torch_interop_threads"] == 1
    assert all(value == "1" for value in policy["requested_environment"].values())
    assert all(pool["num_threads"] == 1 for pool in policy["native_pools"])
    assert set(policy["configuration_origins"]) == {"numpy", "scipy", "torch"}
    if policy["accelerate"] is not None:
        assert policy["accelerate"]["BLASGetThreading"] == 1
        assert policy["accelerate"]["scope"] == "calling thread"


@pytest.mark.parametrize("tuning", [False, True])
def test_post_operation_thread_violation_aborts_instead_of_continuing_candidates(
    monkeypatch, tuning
):
    pytest.importorskip("optuna")
    from benchmarks.rc_bench import budget

    class Adapter:
        calls = 0

        def fit(self, *args):
            self.calls += 1
            raise RuntimeError("ordinary fit failure")

    adapter = Adapter()
    checks = iter((None, _threads.ThreadPolicyError("changed native limit")))

    def verify():
        outcome = next(checks)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(budget, "verify_thread_policy", verify)
    with pytest.raises(_threads.ThreadPolicyError, match="changed native"):
        protocol = budget.Protocol("lorenz", 5, 1, 1, 0.02, 0.9, (5,), (6,))
        if tuning:
            study = budget.optuna.create_study(direction="maximize")
            budget.tune_track(
                adapter, budget.np.ones((6, 3)), protocol, study, 10, smoke=True, max_trials=2
            )
        else:
            budget.evaluate_candidate(
                adapter, budget.np.ones((6, 3)), None, protocol, (7,), (5,), keep_errors=True
            )
    assert adapter.calls == 1


@pytest.mark.parametrize("operation", ["fit", "forecast"])
def test_direct_adapter_policy_errors_escape_final_evaluation(operation):
    pytest.importorskip("optuna")
    from benchmarks.rc_bench import budget

    class Adapter:
        def fit(self, *args):
            if operation == "fit":
                raise _threads.ThreadPolicyError("adapter policy violation")
            return self, {}

        def forecast(self, *args):
            raise _threads.ThreadPolicyError("adapter policy violation")

    with pytest.raises(_threads.ThreadPolicyError, match="adapter policy"):
        budget.evaluate_candidate(
            Adapter(),
            budget.np.ones((6, 3)),
            None,
            budget.Protocol("lorenz", 5, 1, 1, 0.02, 0.9, (5,), (6,)),
            (7,),
            (5,),
            keep_errors=True,
        )
