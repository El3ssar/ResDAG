# Under the hood

The guides tell you *what to call*; this section tells you *what the library
computes*. Every equation here maps one-to-one onto the code, with pointers to
the classes that implement it.

| Page | Covers |
| --- | --- |
| [Reservoir equations](reservoir-equations.md) | The leaky-ESN state update, spectral radius, leak rate, bias, and the NG-RC feature map |
| [Readout fitting](readout-fitting.md) | The ridge-regression objective, intercept handling, and the conjugate-gradient solver |
| [Timing & alignment](timing-and-alignment.md) | The one-step-ahead convention, warmup semantics, the autoregressive forecast loop, and driver alignment |

If you are validating ResDAG against your own derivations or another
implementation, read these three pages first — they pin down every convention
(index origins, shift directions, what is and isn't centered) that papers
usually leave implicit.
