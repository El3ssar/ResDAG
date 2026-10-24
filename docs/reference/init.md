---
description: API reference for resdag.init — topology initializers, input/feedback initializers, matrix builders, and spec resolvers.
---

<span class="nb-kicker">Reference</span>

# Initialization

How frozen weights get their values: topology initializers for the square
recurrent matrix, input/feedback initializers for the rectangular ones, the
registries behind both, and the resolvers that accept names, tuples,
callables, or configured objects interchangeably.

Use `recurrent_initializer=` for recurrent graph or matrix weights and
`feedback_initializer=` / `input_initializer=` for rectangular projections.
These configure frozen weights; the reservoir cell still determines the state
update equation. Registry decorators, initializer base classes and callable
specifications remain supported extension interfaces. See
[custom components](../advanced/custom-components.md) for complete recipes.

## Recurrent initializers

::: resdag.init.recurrent
    options:
      members:
        - get_recurrent_initializer
        - register_recurrent_initializer
        - show_recurrent_initializers

## Topologies and custom builders

::: resdag.init.topology
    options:
      members:
        - TopologyInitializer
        - GraphTopology
        - MatrixTopology
        - get_topology
        - register_graph_topology
        - register_matrix_topology
        - estimate_spectral_radius
        - scale_to_spectral_radius
        - show_topologies

---

## Input and feedback initializers

::: resdag.init.input_feedback
    options:
      members:
        - InputFeedbackInitializer
        - FunctionInitializer
        - InformedInputInitializer
        - LogisticMappingInitializer
        - MinimalInputInitializer
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
        - WeightedInputInitializer
        - WeightedMinimalInitializer
        - ZeroInitializer

---

## Direct matrix builders

::: resdag.init.matrices
    options:
      members:
        - antisymmetric_matrix
        - band
        - block_diagonal
        - cycle_jumps
        - delay_line
        - delay_line_backward
        - fast_spectral_initialization
        - orthogonal_matrix
        - self_loop_cycle
        - simple_cycle
        - singular_spectrum
        - eigen_spectrum
        - signed_permutation
        - householder_orthogonal
        - toeplitz

## Specification types and resolution

Resolvers preserve each initializer's supported seed and device conventions.
`select_recurrent_spec` accepts either the original `topology` spelling or
`recurrent_initializer` and rejects supplying both.

::: resdag.init.utils
    options:
      members:
        - resolve_topology
        - resolve_initializer
        - resolve_recurrent_initializer
        - select_recurrent_spec
        - TopologySpec
        - InitializerSpec
        - RecurrentInitializerSpec
