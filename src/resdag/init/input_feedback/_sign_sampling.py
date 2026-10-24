"""Deterministic and stochastic sign-sampling rules (RC.jl parity).

This module centralizes the sign-assignment rules that ReservoirComputing.jl
(RC.jl) exposes as its `sampling_type` keyword for the minimal-complexity
input initializers of Rodan & Tiňo. Reproduced from RC.jl
`src/inits/inits_components.jl` (`no_sample` / `bernoulli_sample!` /
`irrational_sample!` / `_regular_sample!`).

The one subtlety worth stating up front: RC.jl matrices are **column-major**
(Julia), and every sampler walks a matrix in `eachindex` order — i.e. column 0
top-to-bottom, then column 1, and so on. The deterministic samplers
(`irrational`/`regular`) key their per-entry decision on that flat traversal
position, so a faithful reproduction must apply the flat sign vector in
**column-major (Fortran) order**. All helpers here return a *flat* sign vector of
length `rows * cols` in that column-major order; the initializers reshape it
with `order="F"` before multiplying.

Notes
-----
RC.jl's docstring *example outputs* for these samplers are RNG-/version
dependent (the default sampler is `:bernoulli_sample!`, and one docstring even
names a non-existent `:irrational` alias), so they are **not** a reliable
oracle. These helpers reproduce the *source algorithm* verbatim instead, which
is the stable parity target.

References
----------
A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network," IEEE Trans.
Neural Netw., vol. 22, no. 1, pp. 131-144, Jan. 2011.
"""

from __future__ import annotations

import numpy as np

# Sign rules accepted by the minimal-complexity input initializers, matching the
# ``sampling_type`` values of RC.jl (with the trailing ``!`` and ``_sample``
# dropped for a friendlier Python surface: "bernoulli", "irrational", "regular",
# "none").
_VALID_SIGN_RULES = ("none", "bernoulli", "irrational", "regular")


def _pi_digits(count: int) -> np.ndarray:
    """First `count` digits of pi with the decimal point removed.

    Returns `[3, 1, 4, 1, 5, 9, ...]` — the same `ir_array` RC.jl builds via
    `string(BigFloat(pi))` then stripping the `.`. Computed on demand with a
    self-contained integer spigot (no external dependency), so an arbitrarily
    large matrix always has enough digits (RC.jl likewise raises its BigFloat
    precision to fit the matrix).

    The spigot yields the leading `3` then the fractional digits, so the
    returned array is exactly RC.jl's 1-indexed `ir_array` shifted to 0-indexed
    NumPy.
    """
    digits: list[int] = []
    q, r, t, k, n, length = 1, 0, 1, 1, 3, 3
    while len(digits) < count:
        if 4 * q + r - t < n * t:
            digits.append(n)
            q, r, t, k, n, length = (
                10 * q,
                10 * (r - n * t),
                t,
                k,
                (10 * (3 * q + r)) // t - 10 * n,
                length,
            )
        else:
            q, r, t, k, n, length = (
                q * k,
                (2 * q + r) * length,
                t * length,
                k + 1,
                (q * (7 * k + 2) + r * length) // (t * length),
                length + 2,
            )
    return np.array(digits[:count], dtype=np.int64)


def _validate_sign_rule(sign_rule: str) -> str:
    """Validate and normalize a `sign_rule` name."""
    if sign_rule not in _VALID_SIGN_RULES:
        valid = ", ".join(repr(r) for r in _VALID_SIGN_RULES)
        raise ValueError(f"sign_rule must be one of {valid}; got {sign_rule!r}.")
    return sign_rule


def _bernoulli_signs(total: int, positive_prob: float, rng: np.random.Generator) -> np.ndarray:
    """RC.jl `bernoulli_sample!`: each entry is negated with prob `1 - positive_prob`.

    RC.jl flips the sign when `rand(rng) > positive_prob`, so an entry stays
    positive with probability `positive_prob`.
    """
    draws = rng.random(total)
    signs = np.ones(total, dtype=np.int64)
    signs[draws > positive_prob] = -1
    return signs


def _irrational_signs(total: int, start: int) -> np.ndarray:
    """RC.jl `irrational_sample!` (pi): negate where the pi digit is odd.

    Reproduces the source exactly: for element `element_index = 1..total` the
    decision digit is `ir_array[start + element_index]` (Julia 1-indexed into
    the digit string). The entry is negated when that digit is odd.
    """
    digit_indices = start + np.arange(1, total + 1)  # Julia 1-indexed positions
    max_needed = int(digit_indices.max())
    pi = _pi_digits(max_needed)
    # Julia ir_array[j] == pi[j - 1] (0-indexed here).
    chosen = pi[digit_indices - 1]
    signs = np.where(chosen % 2 == 1, -1, 1).astype(np.int64)
    return signs


def _regular_signs(total: int, strides: int) -> np.ndarray:
    """RC.jl `_regular_sample!` (integer strides): negate every `strides`-th entry.

    RC.jl negates entry `idx` (1-indexed flat position) when
    `idx % strides == 0`.
    """
    idx = np.arange(1, total + 1)
    signs = np.where(idx % strides == 0, -1, 1).astype(np.int64)
    return signs


def build_sign_matrix(
    rows: int,
    cols: int,
    sign_rule: str,
    *,
    positive_prob: float = 0.5,
    start: int = 1,
    strides: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a `(rows, cols)` matrix of `{-1, +1}` signs per a RC.jl rule.

    The flat sign vector is generated in RC.jl's column-major (`eachindex`)
    traversal order and reshaped with `order="F"` so the per-entry decisions
    land on the same matrix positions as RC.jl.

    Parameters
    ----------
    rows, cols : int
        Matrix shape `(reservoir_size, in_size)`.
    sign_rule : {"none", "bernoulli", "irrational", "regular"}
        The RC.jl `sampling_type` (without the `_sample!` suffix). `"none"`
        leaves every sign `+1`.
    positive_prob : float, default=0.5
        Probability an entry stays positive under `"bernoulli"` (RC.jl
        `positive_prob`).
    start : int, default=1
        First pi-digit offset for `"irrational"` (RC.jl `start`).
    strides : int, default=2
        Stride for `"regular"` (RC.jl `strides`); every `strides`-th flat
        entry is negated.
    rng : numpy.random.Generator, optional
        Generator for `"bernoulli"`. Defaults to a fresh `default_rng()`.

    Returns
    -------
    numpy.ndarray
        `(rows, cols)` array of `+1` / `-1` (`int64`).
    """
    sign_rule = _validate_sign_rule(sign_rule)
    total = rows * cols

    if sign_rule == "none":
        flat = np.ones(total, dtype=np.int64)
    elif sign_rule == "bernoulli":
        if rng is None:
            rng = np.random.default_rng()
        flat = _bernoulli_signs(total, positive_prob, rng)
    elif sign_rule == "irrational":
        flat = _irrational_signs(total, start)
    else:  # "regular"
        flat = _regular_signs(total, strides)

    # RC.jl walks the matrix column-major, so reshape the flat sign vector in
    # Fortran order to place each decision on the RC.jl-equivalent entry.
    return flat.reshape((rows, cols), order="F")
