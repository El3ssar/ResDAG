---
description: API reference for resdag.init — topology initializers, input/feedback initializers, matrix builders, and spec resolvers.
---

<span class="nb-kicker">Reference</span>

# Initialization

How frozen weights get their values: topology initializers for the square
recurrent matrix, input/feedback initializers for the rectangular ones, the
registries behind both, and the resolvers that accept names, tuples,
callables, or configured objects interchangeably.

::: resdag.init.recurrent
    options:
      members:
        - get_recurrent_initializer
        - register_recurrent_initializer
        - show_recurrent_initializers

::: resdag.init.topology
    options:
      members:
        - TopologyInitializer
        - GraphTopology
        - MatrixTopology
        - get_topology
        - register_graph_topology
        - register_matrix_topology
        - scale_to_spectral_radius
        - show_topologies

---

::: resdag.init.input_feedback
    options:
      members:
        - InputFeedbackInitializer
        - FunctionInitializer
        - get_input_feedback
        - register_input_feedback
        - show_input_initializers
        - BernoulliInputInitializer
        - BinaryBalancedInitializer
        - ChainOfNeuronsInputInitializer
        - ChebyshevInitializer
        - ChessboardInitializer
        - DendrocycleInputInitializer
        - NormalInputInitializer
        - OppositeAnchorsInitializer
        - PseudoDiagonalInitializer
        - RandomBinaryInitializer
        - RandomInputInitializer
        - RingWindowInputInitializer
        - UniformInputInitializer
        - ZeroInitializer

---

::: resdag.init.matrices
    options:
      members:
        - orthogonal_matrix
        - singular_spectrum
        - eigen_spectrum
        - signed_permutation
        - householder_orthogonal

::: resdag.init.utils
    options:
      members:
        - resolve_topology
        - resolve_initializer
        - TopologySpec
        - InitializerSpec
