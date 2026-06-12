---
description: BibTeX for ResDAG and the reservoir-computing papers behind each method.
---

<span class="rd-eyebrow">About</span>

# Citation

Using ResDAG in a paper? Cite the software once, then cite the method you
actually used — ESN, NG-RC, Ott-style augmentation — from the table below.

## Software

```bibtex
@software{resdag2024,
  author = {Estevez-Moya, Daniel},
  title = {ResDAG: PyTorch-native reservoir computing},
  url = {https://github.com/El3ssar/resdag},
  version = {0.4.0},
  year = {2024},
}
```

Set `version` and `year` to the release you actually used.

## References by method

| Feature in ResDAG | Reference |
|-------------------|-----------|
| Echo State Networks | Jaeger, *The “echo state” approach to analysing and training recurrent neural networks* (2001) |
| Reservoir computing survey | Lukoševičius & Jaeger, *Reservoir computing approaches to recurrent neural network training* (2009) |
| NG-RC | Gauthier et al., *Next Generation Reservoir Computing* (2021) |
| Ott ESN / chaos | Pathak et al., *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach* (2018) |
