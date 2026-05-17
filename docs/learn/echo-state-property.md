# The Echo State Property

!!! info "Why this exists"
    A reservoir must **forget** perturbed initial conditions under the same input drive.
    Without this *echo state property* (ESP), state trajectories depend on hidden
    history you never observe, and readout fitting becomes ill-posed.

## Definition (informal)

Fix an input sequence $u_{1:T}$. The reservoir has the ESP if two runs that start from
different initial states $x_0 \neq x_0'$ but receive the same inputs produce states that
**converge**:

$$
\| x_t - x_t' \| \rightarrow 0 \quad \text{as } t \rightarrow \infty
$$

Intuitively: the input "washes out" memory of the starting state.

## Spectral radius rule of thumb

For standard tanh ESNs, a practical stability knob is the **spectral radius** $\rho$ of
$W_{\mathrm{res}}$ (largest absolute eigenvalue, after scaling the matrix):

| Regime | Typical $\rho$ | Behaviour |
|--------|----------------|-----------|
| Contractive | $\rho < 1$ | Strong fading memory; ESP usually holds |
| Edge of chaos | $\rho \approx 1$ | Rich dynamics, still often stable with leak rate $< 1$ |
| Expanding | $\rho \gg 1$ | States blow up; ESP fails |

In ResDAG you set this when building the layer:

```python
from resdag.layers import ESNLayer

reservoir = ESNLayer(
    reservoir_size=500,
    feedback_size=3,
    spectral_radius=0.9,  # scale W_res so ρ ≈ 0.9
)
```

Topology initialization builds an unscaled graph adjacency, then rescales to the requested
`spectral_radius` (see [Graph topologies](topologies.md)).

## Measuring ESP in ResDAG

Use [`esp_index`](../reference/utils/states.md) to estimate how quickly state differences
decay under a given drive:

```python
from resdag.utils.states import esp_index

indices = esp_index(model, feedback_seq=train_feedback, iterations=10)
print(indices)  # per ESNLayer name → scalar list
```

The helper:

1. Runs the model from **zero** initial state and records reservoir outputs.
2. Repeats with **random** initial states.
3. Averages $\|x_t^{\mathrm{base}} - x_t^{\mathrm{rand}}\|$ over batches and iterations.

Lower values mean faster convergence (better ESP). With **linear** activation (`identity`),
input contributions cancel in the difference dynamics — small distances are expected and
not a bug.

## See also

- [Reservoir layers](reservoir-layers.md)
- [`ESNLayer`](../reference/layers/reservoirs.md#resdag.layers.reservoirs.ESNLayer)
- [Chaotic forecasting guide](../guides/chaotic-systems.md)
