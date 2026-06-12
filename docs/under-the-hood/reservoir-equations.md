<span class="rd-eyebrow">Under the hood</span>

# Reservoir equations

## The leaky-ESN state update

`ESNCell` (wrapped by `ESNLayer`) implements the standard leaky-integrator
Echo State Network update (Jaeger 2001; Lukoševičius 2012, Eq. 2–3):

$$
\tilde{h}_t = f\!\left(W_{fb}\, x^{fb}_t + W_{in}\, x^{in}_t + W_{rec}\, h_{t-1} + b\right)
$$

$$
h_t = (1 - \alpha)\, h_{t-1} + \alpha\, \tilde{h}_t
$$

| Symbol | Code | Shape | Notes |
| --- | --- | --- | --- |
| $h_t$ | `layer.state` | `(batch, reservoir_size)` | Carried across `forward` calls |
| $W_{fb}$ | `weight_feedback` | `(reservoir_size, feedback_size)` | Default $\mathcal{U}(-1, 1)$, or any [initializer](../cookbook/initializers.md) |
| $W_{in}$ | `weight_input` | `(reservoir_size, input_size)` | Only when `input_size` is set (driving inputs) |
| $W_{rec}$ | `weight_hh` | `(reservoir_size, reservoir_size)` | Default $\mathcal{U}(-1, 1)$, or any [graph topology](../cookbook/topologies.md) |
| $b$ | `bias_h` | `(reservoir_size,)` | $\mathcal{U}(-\beta, \beta)$ with $\beta$ = `bias_scaling` |
| $f$ | `activation` | — | `tanh` (default), `relu`, `sigmoid`, `identity` |
| $\alpha$ | `leak_rate` | — | $1.0$ = no leaking (plain RNN update) |

The activation wraps **only the pre-activation** — not the leaky combination.
With $\alpha = 1$ the update reduces to
$h_t = f(W_{fb} x^{fb}_t + W_{in} x^{in}_t + W_{rec} h_{t-1} + b)$.

## Spectral radius

`spectral_radius=ρ` rescales $W_{rec}$ so that its largest absolute
eigenvalue equals $\rho$:

$$
W_{rec} \leftarrow \rho \, \frac{W_{rec}}{\max_i |\lambda_i(W_{rec})|}
$$

$\rho < 1$ is the classic (sufficient-ish, not necessary) heuristic for the
*echo state property* — the requirement that the reservoir state becomes
independent of its initial condition after enough driven steps, so that two
runs over the same data converge to the same trajectory. Larger $\rho$
lengthens memory and pushes the dynamics toward instability; values between
$0.8$ and $1.5$ are all usable depending on the task and leak rate. Use
`resdag.utils.states.esp_index` to measure state convergence empirically.

## Leak rate

$\alpha \in (0, 1]$ low-pass filters the dynamics: the reservoir time scale
is roughly $1/\alpha$ steps. Slow, smooth signals (relative to the sampling
rate) benefit from small $\alpha$; chaotic systems sampled near their natural
time scale usually run at $\alpha = 1$.

## Bias

The bias is a **fixed random vector**, drawn once at construction from
$\mathcal{U}(-\beta, \beta)$ where $\beta$ = `bias_scaling` (default `1.0`,
matching the default scale of $W_{fb}$). It is frozen together with the other
reservoir weights on the standard (non-trainable) path.

Its job is symmetry breaking. `tanh` is odd, so a bias-free reservoir
satisfies

$$
h_t(-x_{1:t}) = -h_t(x_{1:t})
$$

for zero-initialized state: negate the input series and every state negates
with it. The readout then sees a feature set constrained to odd functions of
the input, which measurably hurts tasks whose target has an even component.
A per-neuron random offset removes the constraint and diversifies the
operating points of the neurons (some saturate earlier, some stay near the
linear regime).

Knobs:

- `bias=False` — no bias term at all.
- `bias_scaling=0.0` — keeps the parameter but zero-valued. This reproduces
  the behaviour of ResDAG ≤ 0.4, where the bias was zero-initialized and
  therefore inert. Checkpoints saved with 0.4 load cleanly either way since
  the parameter shape is unchanged.

!!! warning "Changed in 0.5"
    Before 0.5 `bias=True` created a zero vector that was then frozen — a
    no-op. Results with default settings change after the fix (generally for
    the better on forecasting benchmarks). Pin `bias_scaling=0.0` to
    reproduce old runs exactly.

## NG-RC feature map

`NGReservoir` (wrapping `NGCell`) has **no weights and no recurrence**
(Gauthier et al. 2021). Its "state" is a FIFO buffer of past inputs, and its
output is a deterministic feature vector:

$$
O_{lin,t} = x_t \oplus x_{t-s} \oplus \cdots \oplus x_{t-(k-1)s}
\qquad (D = k \cdot d \text{ entries})
$$

$$
O_{nonlin,t} = \left\{\, \textstyle\prod_{j \in J} O_{lin,t}[j] \;\middle|\; J \in \binom{\{1..D\}+\text{repl.}}{p} \right\}
$$

$$
O_t = [\,1\,] \oplus O_{lin,t} \oplus O_{nonlin,t}
$$

with $k$ delay taps spaced $s$ steps apart and all degree-$p$ monomials with
replacement, $\binom{D+p-1}{p}$ of them. The optional constant $1$ plays the
role of the readout intercept. The buffer needs $(k-1)s$ steps to fill —
discard that many initial outputs.

## References

- H. Jaeger, *The "echo state" approach to analysing and training recurrent
  neural networks*, GMD Report 148 (2001).
- M. Lukoševičius, *A Practical Guide to Applying Echo State Networks*,
  Neural Networks: Tricks of the Trade (2012).
- D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, *Next generation
  reservoir computing*, Nat. Commun. 12, 5564 (2021).
