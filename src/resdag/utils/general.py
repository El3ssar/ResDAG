"""General utility functions for resdag."""

import random

import numpy as np
import torch

# Anything accepted as a reproducibility seed throughout resdag.  A plain
# ``int`` pins a value; a ``torch.Generator`` lets callers (e.g. an HPO loop)
# carry a per-trial generator; ``None`` defers to the global RNG.
SeedLike = int | torch.Generator | None

# Anything accepted as a device spec by :func:`resolve_device`.  ``'auto'``
# (or ``None``) selects the best available backend; an explicit string
# (``'cpu'``, ``'cuda'``, ``'cuda:1'``, ``'mps'``, …) or a :class:`torch.device`
# passes straight through.
DeviceLike = str | torch.device | None


def coerce_seed_to_int(seed: SeedLike) -> int | None:
    """Reduce a torch/int/None seed to a plain ``int`` seed (or ``None``).

    The NumPy-backed graph builders and named initializers thread an integer
    (or ``None``) seed down to :func:`create_rng`.  A :class:`torch.Generator`
    cannot be passed there directly, so this helper extracts a deterministic
    integer from its initial seed — two generators created with the same
    ``manual_seed`` therefore yield the same int, keeping the topology a pure
    function of the generator.

    Parameters
    ----------
    seed : int, torch.Generator, or None
        Seed in any of the accepted forms.

    Returns
    -------
    int or None
        ``seed`` itself if it was an ``int``; the generator's
        ``initial_seed()`` if it was a :class:`torch.Generator`; ``None`` if it
        was ``None``.
    """
    if seed is None or isinstance(seed, int):
        return seed
    if isinstance(seed, torch.Generator):
        # initial_seed() is the value the generator was seeded with; identical
        # generators (same manual_seed) therefore map to the same int.
        return int(seed.initial_seed())
    raise TypeError(
        f"Invalid seed type: {type(seed).__name__}. Expected int, torch.Generator, or None."
    )


def create_torch_generator(
    seed: SeedLike = None,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Create a torch :class:`~torch.Generator` for reproducible weight draws.

    Used for the default ``nn.init`` weight/bias draws inside reservoir cells so
    that a single ``seed`` deterministically fixes every parameter without
    touching (and without depending on the prior state of) torch's global RNG.

    Parameters
    ----------
    seed : int, torch.Generator, or None
        If ``int``, seeds a fresh generator on ``device``.
        If ``torch.Generator``, it is returned as-is (the caller's generator is
        threaded straight through, so successive draws advance one stream).
        If ``None``, the generator is seeded from torch's global RNG so that
        ``torch.manual_seed`` still propagates — matching :func:`create_rng`.

    device : torch.device, str, or None
        Device for the new generator.  Ignored when ``seed`` is already a
        :class:`torch.Generator`.

    Returns
    -------
    torch.Generator
        A generator suitable for passing to ``nn.init.*`` via ``generator=``.

    Notes
    -----
    When ``seed is None`` the integer seed is sampled from torch's global RNG
    via ``torch.randint``, advancing the global state but tying the draws to
    ``torch.manual_seed`` for end-to-end reproducibility.
    """
    if isinstance(seed, torch.Generator):
        return seed
    generator = torch.Generator(device=device if device is not None else "cpu")
    if seed is None:
        seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    generator.manual_seed(int(seed))
    return generator


def create_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Create a NumPy random number generator.

    This is the NumPy-RNG-for-graph-topology helper: it is threaded into the
    graph builders and named initializers, which draw exclusively from NumPy.
    For whole-program reproducibility (torch + NumPy + ``random`` at once) use
    :func:`seed_everything` instead; for a torch :class:`~torch.Generator` for
    weight/bias draws use :func:`create_torch_generator`.

    Parameters
    ----------
    seed : int, np.random.Generator, or None
        If int, used as seed for a new Generator.
        If Generator, returned as-is.
        If None, the seed is derived from torch's global RNG so that
        ``torch.manual_seed`` propagates to the generator. This keeps
        graph-based topologies (which draw from NumPy) reproducible under
        the same torch global seed, matching the behaviour of matrix
        topologies that draw from torch directly.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator.

    Notes
    -----
    When ``seed is None`` the numpy seed is sampled from torch's global RNG
    via ``torch.randint``. This advances the torch global RNG state, but ties
    every NumPy draw to ``torch.manual_seed`` for end-to-end reproducibility.

    See Also
    --------
    seed_everything : Seed torch, NumPy, and ``random`` in one call.
    create_torch_generator : Build a torch ``Generator`` for weight draws.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        # Derive the numpy seed from torch's global RNG so that
        # torch.manual_seed(...) makes graph generators reproducible.
        seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    return np.random.default_rng(seed)


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducible runs.

    Seeds every global RNG resdag and its dependencies draw from in one call:
    the :mod:`random` module, NumPy's legacy global RNG, torch's CPU RNG, and
    (when available) every visible CUDA device. The seed is returned so callers
    can log or thread it (e.g. an HPO loop deriving a per-trial seed).

    Parameters
    ----------
    seed : int, default 42
        Seed applied to every RNG.
    deterministic : bool, keyword-only, default False
        If ``True``, also request deterministic algorithms from torch by
        enabling :func:`torch.use_deterministic_algorithms` and disabling the
        cuDNN benchmark autotuner. This favours reproducibility over speed and
        may raise at runtime if a non-deterministic op has no deterministic
        implementation, so it is opt-in.

    Returns
    -------
    int
        The ``seed`` that was applied, for logging or downstream threading.

    Notes
    -----
    This seeds the *global* RNGs. For local, side-effect-free reproducibility,
    prefer the generator factories: :func:`create_rng` (NumPy, for graph
    topologies) and :func:`create_torch_generator` (torch, for weight draws).

    Examples
    --------
    >>> from resdag.utils import seed_everything
    >>> seed_everything(0)
    0
    >>> import torch
    >>> seed_everything(0); a = torch.randn(3)
    0
    >>> seed_everything(0); b = torch.randn(3)
    0
    >>> bool(torch.equal(a, b))
    True

    See Also
    --------
    create_rng : NumPy generator for graph topologies (local RNG).
    create_torch_generator : torch generator for weight draws (local RNG).
    resolve_device : Resolve a device spec to a ``torch.device``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def resolve_device(spec: DeviceLike = "auto") -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    Provides the ``'auto'`` convenience used by HPO, examples, and reproducible
    experiments: it picks the best available backend (CUDA, then MPS, then CPU).
    Explicit specs pass through, with a CPU fallback if the requested
    accelerator is unavailable.

    Parameters
    ----------
    spec : str, torch.device, or None, default ``'auto'``
        ``'auto'`` (or ``None``) selects ``cuda`` → ``mps`` → ``cpu`` by
        availability. A :class:`torch.device` is returned unchanged. An explicit
        string (``'cpu'``, ``'cuda'``, ``'cuda:1'``, ``'mps'``, …) is parsed into
        a :class:`torch.device`; if it names an unavailable accelerator the
        result falls back to ``cpu``.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``str``, :class:`torch.device`, or ``None``.

    Examples
    --------
    >>> from resdag.utils import resolve_device
    >>> resolve_device("cpu")
    device(type='cpu')
    >>> resolve_device("auto").type in {"cuda", "mps", "cpu"}
    True

    See Also
    --------
    seed_everything : Seed torch, NumPy, and ``random`` in one call.
    """
    if isinstance(spec, torch.device):
        return spec
    if spec is None or spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if not isinstance(spec, str):
        raise TypeError(
            f"Invalid device spec: {type(spec).__name__}. Expected str, torch.device, or None."
        )
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device
