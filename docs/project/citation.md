---
description: How to cite ResDAG, and the original papers behind each method the library implements.
---

<span class="nb-kicker">Project</span>

# Citation

If ResDAG contributes to published work, cite the software:

```bibtex
@software{estevezmoya_resdag_2026,
  author  = {Estevez-Moya, Daniel},
  title   = {ResDAG: Reservoir computing for PyTorch},
  year    = {2026},
  url     = {https://github.com/El3ssar/ResDAG},
  version = {{{ resdag_version }}},
}
```

---

## Citing the methods

ResDAG implements published methods. Alongside the software entry, cite the
original papers for the methods your work uses.

| You used | Cite |
| -------- | ---- |
| Echo State Networks (`ESNLayer`, any premade model) | Jaeger, *The "echo state" approach to analysing and training recurrent neural networks*, GMD Report 148 (2001) |
| ESN design and tuning practice | Lukoševičius, *A Practical Guide to Applying Echo State Networks*, in Neural Networks: Tricks of the Trade, Springer (2012) |
| State-augmented chaos architecture (`ott_esn`, `power_augmented`) | Pathak, Hunt, Girvan, Lu & Ott, *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach*, Phys. Rev. Lett. 120, 024102 (2018) |
| Next-generation reservoir computing (`NGReservoir`, `NGCell`) | Gauthier, Bollt, Griffith & Barbosa, *Next generation reservoir computing*, Nat. Commun. 12, 5564 (2021) |

!!! note "Attribution"
    The squared-state architecture is colloquially called the "Ott ESN" —
    the citable source is Pathak et al. (2018), where Ott is the senior
    author, not a paper by Ott alone.
