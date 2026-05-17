# NG-RC vs ESN

Side-by-side on the same 3D trajectory task. NG-RC needs no recurrent weights; ESN
needs spectral-radius tuning but scales to richer memory.

## Shared data

```python
import torch

def make_data(n=2000):
    t = torch.linspace(0, 50, n)
    x = torch.stack([torch.sin(t), torch.cos(0.7 * t), torch.sin(0.3 * t)], dim=-1)
    return (x - x.mean(0)) / x.std(0)

data = make_data().unsqueeze(0)
warmup, train = data[:, :200], data[:, 200:800]
target = data[:, 201:801]
f_warmup = data[:, 800:900]
```

## ESN path

```python
from resdag import ott_esn
from resdag.training import ESNTrainer

esn = ott_esn(300, feedback_size=3, output_size=3, spectral_radius=0.9)
ESNTrainer(esn).fit((warmup,), (train,), {"output": target})
esn.reset_reservoirs()
pred_esn = esn.forecast(f_warmup, horizon=100)
```

## NG-RC path

```python
import pytorch_symbolic as ps
from resdag.core import ESNModel
from resdag.layers import NGReservoir
from resdag.layers.readouts import CGReadoutLayer

inp = ps.Input((1, 3))
ngrc = NGReservoir(input_dim=3, k=2, s=1, p=2)(inp)
readout = CGReadoutLayer(ngrc.cell.feature_dim, 3, name="output")(ngrc)
ng_model = ESNModel(inp, readout)

ESNTrainer(ng_model).fit((warmup,), (train,), {"output": target})
# Discard first (k-1)*s steps after forward if evaluating train error
```

## Comparison

| | ESN | NG-RC |
|---|-----|-------|
| Recurrent weights | Yes | No |
| Main knobs | `reservoir_size`, `spectral_radius`, topology | `k`, `s`, `p` |
| Feature size | ≈ reservoir size | combinatorial in `p` |
| GPU memory | States `(B,T,N)` | Features can explode |

## See also

- [NG-RC concept](../learn/ngrc.md)
- [`NGReservoir`](../reference/layers/reservoirs.md)
