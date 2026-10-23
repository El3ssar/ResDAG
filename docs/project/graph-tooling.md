---
description: Why functional construction stays, and how standard PyTorch graph tooling fits around it.
---

# Graph tooling and PyTorch integration

ResDAG keeps `layer(input)` functional construction. The common `rd.Model`
owns the original graph once, so optimizers, device moves, hooks and state
dictionaries see ordinary PyTorch modules. Users do not need private graph
attributes to build, fit or run a model.

## Which tool does which job?

| Tool | Useful role | Current boundary |
| --- | --- | --- |
| `pytorch_symbolic` | Declare branches, shared nodes and named model boundaries with `layer(input)` syntax. | Retained construction backend. ResDAG validates its execution plans internally. |
| PyTorch `nn.Module` | Own layers, parameters and buffers; compose trained models inside larger networks. | Supported by `rd.Model`, with carried recurrent state and original-unit predictions. |
| PyTorch FX | Capture or transform the surrounding network while keeping the reservoir as one module call. | Use a leaf tracer, shown below. Direct tracing through the entire workflow is unsupported. |
| `torch.export` | Produce a tensor computation graph for deployment when a module satisfies export's constraints. | No general ResDAG export guarantee. A stateful workflow, its fitting logic and its Python metadata are not a deployment graph contract. |
| Graphviz/DOT | Render the declared architecture. | Unified `Model.save`/`ESN.save` exports SVG/PNG/PDF/DOT with named ports, preprocessing and delayed feedback; inspection executes no layers. |

Replacing functional construction with FX would require tracing a computation
before recovering the reservoir-specific ports, fitting stages and state
rules. That would add a second interpretation of the same model. Keep the
current constructor and use standard tools at the boundaries where they
already work.

This is a scoped implementation decision, not a claim that the existing
backend is permanent. A future replacement must preserve functional syntax,
shared-module ownership, multiple heads, fitting order, checkpoint behavior
and explicit recurrent state, with measured benefits.

## Trace a larger PyTorch pipeline

An FX leaf remains a `call_module` operation: FX records the call without
executing or tracing into the reservoir. The captured network shares that
same model instance. The same leaf rule supports the `rd.ESN` preset.
Resetting either reference resets the same reservoir.
The [common model guide](../start/model-workflow.md#capture-the-surrounding-pipeline-with-fx)
contains a complete executable example.

Regression tests verify tensor and named multi-head pipelines, unchanged
state and hooks during capture, parameter identity, prediction/state parity,
upstream gradients, chunked state carry and dtype moves. These checks run on
the minimum supported Torch version as part of the regular suite. They do
not establish compatibility with every third-party graph transformation.

Direct `torch.fx.symbolic_trace(pipeline)` currently reaches the workflow's
concrete tensor validation and fails. A direct strict export probe of a fitted
recurrent pipeline on Torch 2.10 also fails on a tensor with `grad_fn`.
Neither failure justifies silently freezing state or removing validation.
Deployment export needs its own explicit-state contract and acceptance tests.

See PyTorch's [FX documentation](https://docs.pytorch.org/docs/stable/fx.html)
for leaf tracing and control-flow limits, and its
[export programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html)
for graph capture and state constraints.
