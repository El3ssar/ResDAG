---
description: The leaky-ESN update equation, spectral radius scaling, the echo state property, why the bias exists, and the NG-RC feature map — each cross-checked against the code.
---

<span class="nb-kicker">Theory · Dynamics</span>

# Reservoir dynamics

One equation drives every ESN in the library; one feature map drives every
NG-RC. This page states both exactly as the code computes them, then walks
the knobs — spectral radius, leak rate, bias — back to the dynamical
properties they control.

## The leaky-ESN update

`ESNCell` implements the standard leaky-integrator update (Jaeger 2001;
Lukoševičius 2012) in two lines — a nonlinear pre-activation, then a
linear leak:

$$
\tilde h_t = f\!\left(W_{rec}\,h_{t-1} + W_{fb}\,u_t + W_{in}\,d_t + b\right)
$$

$$
h_t = (1-\alpha)\,h_{t-1} + \alpha\,\tilde h_t
$$

The activation $f$ wraps **only** the pre-activation. The leak mixes the
old state with the activated candidate *outside* $f$ — in code,
`torch.lerp(state, new_state, leak_rate)`, which is exactly
$h_{t-1} + \alpha(\tilde h_t - h_{t-1})$. When `leak_rate=1.0` (the
default) the `lerp` is skipped and $h_t = \tilde h_t$: a standard RNN
update.

Symbols map to `ESNCell` attributes one-to-one:

| Symbol | Code | Shape / value |
| --- | --- | --- |
| $h_t$ | `layer.state` | `(batch, reservoir_size)` |
| $u_t$ (feedback) | first `forward` argument | `(batch, feedback_size)` per step |
| $d_t$ (driver) | second `forward` argument, optional | `(batch, input_size)` per step |
| $W_{fb}$ | `weight_feedback` | `(reservoir_size, feedback_size)`, default $\mathcal U(-1,1)$ |
| $W_{in}$ | `weight_input` | `(reservoir_size, input_size)`, `None` without `input_size` |
| $W_{rec}$ | `weight_hh` | `(reservoir_size, reservoir_size)` |
| $b$ | `bias_h` | `(reservoir_size,)`, $\mathcal U(-\beta, \beta)$ with $\beta$ = `bias_scaling` |
| $f$, $\alpha$ | `activation`, `leak_rate` | `"tanh"` default; $\alpha \in [0, 1]$, default 1.0 |

## Spectral radius

After the topology builds $W_{rec}$, the matrix is rescaled so its largest
absolute eigenvalue hits the target:

$$
W_{rec} \leftarrow \frac{\rho_{\text{target}}}{\rho(W_{rec})}\, W_{rec},
\qquad \rho(W) = \max_i |\lambda_i(W)|
$$

(`scale_to_spectral_radius` in `init/topology/base.py`; a matrix with
$\rho \le 10^{-8}$ — the `zero` topology, nilpotent ring structures — is
left unscaled rather than divided by zero.) A bare `ESNLayer` defaults to
`spectral_radius=None`, meaning *no scaling at all*; the premade factories
pass `0.9`.

**The echo state property.** The reservoir is usable when its state
forgets initial conditions: two copies started from different states and
driven by the same input must converge. $\rho < 1$ is the standard
heuristic for this, and a good default — but it is neither necessary nor
sufficient. It guarantees only local stability of the zero-input
linearization; a strongly driven `tanh` reservoir can keep the ESP well
above $\rho = 1$, because the input pushes units into their saturating
region where the effective gain drops. Measure instead of guessing:
`resdag.utils.states.esp_index` runs one orbit from the zero state and
`iterations` orbits from random states $\mathcal U(-1,1)$ under the same
input, and reports $\overline{\lVert h^{\text{base}}_t -
h^{\text{rand}}_t \rVert}$ averaged over time, batch, and restarts. An
index near zero means the trajectories merged — the ESP holds for *your*
signal; a plateau means the reservoir still remembers where it started.

**Leak rate as timescale.** The leak makes forgetting explicit: the
$h_{t-1}$ contribution decays by $(1-\alpha)$ per step, a relaxation time
of $\tau = -1/\ln(1-\alpha) \approx 1/\alpha$ steps. `leak_rate=0.1` gives
the reservoir an intrinsic memory of roughly ten steps — the knob to reach
for when the input evolves much slower than the sampling rate.

---

## Bias and broken symmetry

`tanh` is odd, and so is every term of the bias-free update: with $b = 0$,

$$
h_t(-u, -d \mid -h_{t-1}) = -\,h_t(u, d \mid h_{t-1})
$$

so the entire input-to-state operator satisfies $H(-x) = -H(x)$ — negate
the input sequence and every state trajectory negates exactly. The
reservoir is then structurally blind to the sign of your data: any system
with a mirror symmetry (Lorenz under $(x,y,z) \mapsto (-x,-y,z)$ is the
classic case) gets two attractors mapped onto perfectly antisymmetric
state sets, and forecasts can slip onto the mirror copy. A fixed random
bias $b \sim \mathcal U(-\beta, \beta)$ breaks the identity at every unit,
for free.

!!! warning "Changed in 0.5"
    Before 0.5, `bias=True` allocated a bias that was zero-initialized and
    frozen — effectively no bias at all. Since 0.5 the bias is drawn from
    $\mathcal U(-\beta, \beta)$ with $\beta$ = `bias_scaling` (default
    1.0, matching the feedback/input weight scale). Set
    `bias_scaling=0.0` to reproduce the legacy zero-bias dynamics.

---

## The NG-RC feature map

`NGCell` (Gauthier et al. 2021, Eqs. 5–10) has no weights and no recurrent
dynamics — its "state" is a FIFO buffer of past inputs, and its output is
a deterministic feature vector. For input dimension $d$, $k$ delay taps
spaced $s$ steps apart, and polynomial degree $p$:

**Linear features** — the delay embedding, dimension $D = d\,k$:

$$
O_{\text{lin},i} = \big[\,x_i \,\Vert\, x_{i-s} \,\Vert\, \cdots \,\Vert\, x_{i-(k-1)s}\,\big]
$$

**Nonlinear features** — every degree-$p$ monomial over $O_{\text{lin}}$,
i.e. all multisets of size $p$ from $D$ entries
(`itertools.combinations_with_replacement`), counted by the stars-and-bars
coefficient:

$$
n_{\text{nonlin}} = \binom{D + p - 1}{p}
$$

**Assembly** — constant, linear, and nonlinear blocks concatenated:

$$
O_{\text{total}} = [\,1\,] \oplus O_{\text{lin}} \oplus O_{\text{nonlin}},
\qquad
\dim O_{\text{total}} = \mathbb 1_{\text{const}} + \mathbb 1_{\text{lin}}\,D + \binom{D+p-1}{p}
$$

with the first two blocks present iff `include_constant` /
`include_linear`. The delay buffer holds $(k-1)\,s$ rows and starts
zero-filled, so the first $(k-1)\,s$ outputs mix real data with zeros —
discard them (that is the NG-RC warmup; compare it to the hundreds of
steps an ESN needs). The monomial count explodes combinatorially in $k$,
$p$, and $d$; the cell warns when $\dim O_{\text{total}} > 10{,}000$.

## References

- H. Jaeger, *The "echo state" approach to analysing and training recurrent neural networks*, GMD Report 148 (2001).
- M. Lukoševičius, *A practical guide to applying echo state networks*, in Neural Networks: Tricks of the Trade (2012).
- D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, *Next generation reservoir computing*, Nature Communications 12, 5564 (2021), arXiv:2106.07688.

## Next

[**Readout solvers**](readout.md) — what happens to these states once the
ridge regression sees them.
