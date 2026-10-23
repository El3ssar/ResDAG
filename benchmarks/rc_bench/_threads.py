"""Benchmark-only native thread controls; importing this module is stdlib-only."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "JULIA_NUM_THREADS",
)
_BOOTSTRAPPED = "RESDAG_BENCHMARK_THREAD_BOOTSTRAPPED"
_VERIFY: ContextVar = ContextVar("benchmark_thread_verification", default=None)


class ThreadPolicyError(RuntimeError):
    """A resource-policy violation invalidates the experiment, not one trial."""


def bootstrap_cli() -> None:
    """Restart once so even startup imports see the requested environment."""
    if os.environ.get(_BOOTSTRAPPED) == "1" and all(
        os.environ.get(name) == "1" for name in THREAD_VARIABLES
    ):
        return
    environment = os.environ | {name: "1" for name in THREAD_VARIABLES}
    environment[_BOOTSTRAPPED] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


class _Accelerate:
    """The macOS 15+ BLAS/LAPACK policy is explicitly scoped to this thread."""

    def __init__(self):
        self.path = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        try:
            self.library = ctypes.CDLL(self.path)
            self.setter = self.library.BLASSetThreading
            self.getter = self.library.BLASGetThreading
        except (OSError, AttributeError) as error:
            raise ThreadPolicyError(
                "Cannot verify Accelerate thread policy; macOS 15+ APIs are required"
            ) from error
        self.setter.argtypes, self.setter.restype = [ctypes.c_uint], ctypes.c_int
        self.getter.argtypes, self.getter.restype = [], ctypes.c_uint
        self.previous = int(self.getter())
        if self.setter(1) != 0 or self.getter() != 1:
            raise ThreadPolicyError("Accelerate did not accept BLAS_THREADING_SINGLE_THREADED")

    def restore(self):
        self.setter(self.previous)


def _verify_backends(configurations, pools, accelerate):
    for package, config in configurations.items():
        for kind in ("blas", "lapack"):
            name = str(config.get("Build Dependencies", {}).get(kind, {}).get("name", "")).lower()
            if "accelerate" in name:
                if accelerate is None or accelerate.getter() != 1:
                    raise ThreadPolicyError(
                        f"Cannot verify {package} {kind} Accelerate thread policy"
                    )
            elif not any(
                pool.get("user_api") == "blas"
                and bool(pool.get("internal_api"))
                and str(pool.get("internal_api", "")) in name
                for pool in pools
            ):
                raise ThreadPolicyError(f"Unverified {package} {kind} backend: {name!r}")
    if any(pool.get("num_threads") != 1 for pool in pools):
        raise ThreadPolicyError(
            "A native BLAS/OpenMP thread counter did not accept the limit of one"
        )


def verify_thread_policy() -> None:
    """Recheck an active benchmark policy without changing its settings."""
    verify = _VERIFY.get()
    if verify is not None:
        verify()


@contextmanager
def single_thread_policy():
    """Hold verified native API settings throughout one synchronous benchmark."""
    try:
        import threadpoolctl
    except ImportError as error:
        raise ImportError(
            "Install benchmark dependencies: pip install -r benchmarks/requirements.txt"
        ) from error
    import numpy as np
    import scipy
    import scipy.linalg  # ensure the active SciPy BLAS is loaded before inspection
    import torch

    if threading.current_thread() is not threading.main_thread():
        raise ThreadPolicyError("The verified benchmark must run on the calling main thread")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    configurations = {
        "numpy": np.show_config(mode="dicts"),
        "scipy": scipy.show_config(mode="dicts"),
    }
    torch_config = torch.__config__.show()
    uses_accelerate = (
        "accelerate" in str(configurations).lower() or "accelerate" in torch_config.lower()
    )
    accelerate = _Accelerate() if uses_accelerate else None
    token = None
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            owner = threading.get_ident()

            def verify():
                if threading.get_ident() != owner:
                    raise ThreadPolicyError(
                        "Benchmark execution moved outside its verified calling thread"
                    )
                _verify_backends(configurations, threadpoolctl.threadpool_info(), accelerate)
                if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
                    raise ThreadPolicyError("Torch did not retain one intra-op and inter-op thread")

            verify()
            token = _VERIFY.set(verify)
            metadata = {
                "workers": 1,
                "dtype": "float64",
                "requested_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
                "fresh_cli_bootstrap": os.environ.get(_BOOTSTRAPPED) == "1",
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "native_pools": threadpoolctl.threadpool_info(),
                "build_configurations": configurations,
                "torch_build_configuration": torch_config,
                "torch_parallel_configuration": torch.__config__.parallel_info(),
                "module_origins": {
                    module.__name__: str(Path(module.__file__).resolve())
                    for module in (np, scipy, torch, threadpoolctl)
                },
                "configuration_origins": {
                    module.__name__: str(Path(module.__config__.__file__).resolve())
                    for module in (np, scipy, torch)
                },
                "accelerate": (
                    None
                    if accelerate is None
                    else {
                        "library": accelerate.path,
                        "BLASGetThreading": int(accelerate.getter()),
                        "mode": "BLAS_THREADING_SINGLE_THREADED",
                        "scope": "calling thread",
                    }
                ),
                "verification_boundary": "Observed native API settings on the synchronous calling thread; not an OS/kernel thread-occupancy measurement.",
            }
            yield metadata
            verify()
    finally:
        if token is not None:
            _VERIFY.reset(token)
        if accelerate is not None:
            accelerate.restore()
