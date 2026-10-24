"""Read-only DOT descriptions and standard Graphviz rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from torch import nn

from resdag.core._execution import _symbolic_operations
from resdag.core._workflow_backends import workflow_backend
from resdag.core.model import ESNModel

DIAGRAM_FORMATS = ("svg", "png", "pdf", "dot")


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _shape_label(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "collection/value"
    dimensions = tuple(shape)
    if not dimensions:
        return "scalar"
    if len(dimensions) == 3:
        return f"(batch, time, {dimensions[-1]})"
    return str(dimensions)


class _Diagram:
    def __init__(self, *, show_shapes: bool, show_trainable: bool, rankdir: str) -> None:
        if rankdir not in ("TB", "LR", "BT", "RL"):
            raise ValueError("rankdir must be TB, LR, BT or RL")
        self.shapes = show_shapes
        self.trainable = show_trainable
        self.lines = [
            "digraph ResDAG {",
            f"  rankdir={rankdir};",
            '  graph [bgcolor="white", fontname="Helvetica", pad="0.25"];',
            '  node [shape=box, style="rounded,filled", fillcolor="#eef3fa", fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=10, color="#61738b"];',
        ]

    def node(self, name: str, label: str, *, port: bool = False) -> None:
        self.lines.append(
            f"  {_quoted(name)} [label={_quoted(label)}"
            + (', shape=ellipse, fillcolor="#f5f7fa"' if port else "")
            + "];"
        )

    def edge(self, source: str, target: str, label: str = "", *, feedback: bool = False) -> None:
        attrs = [f"label={_quoted(label)}"]
        if feedback:
            attrs.extend(['style="dashed"', 'color="#b64d36"', "constraint=false"])
        self.lines.append(f"  {_quoted(source)} -> {_quoted(target)} [{', '.join(attrs)}];")

    def module_label(self, name: str, layer: nn.Module) -> str:
        label = f"{name}\n{type(layer).__name__}"
        declared_name = getattr(layer, "name", None)
        if isinstance(declared_name, str) and declared_name and declared_name != name:
            label += f"\nname={declared_name}"
        if isinstance(layer, nn.Sequential):
            label += "\n" + "\n".join(
                f"{path}: {type(child).__name__}"
                for path, child in layer.named_modules(remove_duplicate=False)
                if path and not isinstance(child, nn.Sequential)
            )
        if self.trainable:
            params = tuple(layer.parameters())
            if params:
                label += "\n" + ("trainable" if any(p.requires_grad for p in params) else "frozen")
        return label

    def graph(
        self,
        graph: nn.Module,
        prefix: str,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        input_adapters: tuple[str | None, ...] = (),
        output_adapters: tuple[str | None, ...] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        backend = workflow_backend(graph)
        interface = backend.describe(graph)
        inputs = tuple(f"{prefix}input_{i}" for i in range(len(input_names)))
        outputs = tuple(f"{prefix}output_{i}" for i in range(len(output_names)))
        for kind, names, ids, port_widths in (
            ("Input", input_names, inputs, interface.input_widths),
            ("Output", output_names, outputs, interface.output_widths),
        ):
            for name, node, width in zip(names, ids, port_widths):
                label = f"{kind}: {name}"
                if self.shapes:
                    label += f"\n(batch, time, {width})"
                self.node(node, label, port=True)
        graph_inputs = list(inputs)
        graph_outputs = list(outputs)
        for output, labels in ((False, input_adapters), (True, output_adapters)):
            for index, adapter_label in enumerate(labels):
                if adapter_label is None:
                    continue
                node = f"{prefix}{'output' if output else 'input'}_transform_{index}"
                self.node(node, adapter_label)
                if output:
                    self.edge(node, outputs[index])
                    graph_outputs[index] = node
                else:
                    self.edge(inputs[index], node)
                    graph_inputs[index] = node
        if backend.kind == "symbolic":
            symbolic = cast(ESNModel, graph)
            plan = symbolic._get_execution_plan(require_explicit_state=False)
            producers = {i: (name, "") for i, name in enumerate(graph_inputs)}
            slot_shapes = {
                i: f"(batch, time, {width})" for i, width in enumerate(plan.input_features)
            }
            descriptions = _symbolic_operations(symbolic)
            for index, (op, (node, _, _, siblings)) in enumerate(
                zip(plan.operations, descriptions)
            ):
                op_id = f"{prefix}op_{index}"
                self.node(op_id, self.module_label(op.name, op.layer))
                for parent_index, slot in enumerate(op.parents):
                    source, ordinal = producers[slot]
                    shape = slot_shapes[slot] if self.shapes else ""
                    label = "; ".join(x for x in (ordinal, f"input {parent_index}", shape) if x)
                    self.edge(source, op_id, label)
                nodes = siblings if len(siblings) > 1 else (node,)
                for output_index, (slot, value) in enumerate(zip(op.outputs, nodes)):
                    producers[slot] = (
                        op_id,
                        f"output {output_index}" if len(op.outputs) > 1 else "",
                    )
                    if self.shapes:
                        slot_shapes[slot] = _shape_label(value)
            for target, slot in zip(graph_outputs, plan.output_slots):
                source, ordinal = producers[slot]
                self.edge(source, target, ordinal)
        elif backend.kind == "parallel":
            # Geometry is structural metadata; no patch gathering or forward runs.
            schema = interface.schema
            patches = f"{prefix}patches"
            reservoir, readout, stitch = (
                f"{prefix}{name}" for name in ("reservoir", "readout", "stitch")
            )
            self.node(
                patches, f"Periodic patches\ncore={schema['core_width']}, halo={schema['halo']}"
            )
            self.node(reservoir, self.module_label("shared reservoir", cast(Any, graph).reservoir))
            self.node(readout, self.module_label("shared readout", cast(Any, graph).readout))
            self.node(stitch, "Stitch patch cores")
            for source, target in zip(
                (graph_inputs[0], patches, reservoir, readout, stitch),
                (patches, reservoir, readout, stitch, graph_outputs[0]),
            ):
                self.edge(source, target)
        else:
            ensemble = cast(Any, graph)
            aggregator = f"{prefix}aggregate"
            label = (
                self.module_label("aggregate", ensemble.aggregator_module)
                if ensemble.aggregator_module is not None
                else f"Aggregate: {ensemble._aggregator_str}"
            )
            self.node(aggregator, label)
            for index, member in enumerate(ensemble._iter_models()):
                self.lines.append(f"  subgraph {_quoted(f'cluster_{prefix}{index}')} {{")
                self.lines.append(f"    label={_quoted(f'Member {index}')};")
                member_inputs, member_outputs = self.graph(
                    member, f"{prefix}member_{index}_", input_names, output_names
                )
                self.lines.append("  }")
                for source, target in zip(graph_inputs, member_inputs):
                    self.edge(source, target)
                for source in member_outputs:
                    self.edge(source, aggregator)
            self.edge(aggregator, graph_outputs[0])
        return inputs, outputs


def diagram_dot(
    model: nn.Module,
    *,
    show_shapes: bool = True,
    show_trainable: bool = False,
    rankdir: str = "LR",
) -> str:
    """Describe structure without forwarding, changing modes or opening a viewer."""
    from resdag.core.workflow import Model
    from resdag.facade import ESN

    if isinstance(model, ESN):
        model = model._require_workflow()
    workflow = model if isinstance(model, Model) else None
    graph = workflow.graph if workflow is not None else model
    interface = workflow_backend(graph).describe(graph)
    input_names = (
        workflow.input_names
        if workflow is not None
        else tuple(f"input_{i}" for i in range(len(interface.input_widths)))
    )
    output_names = (
        workflow.output_names
        if workflow is not None
        else tuple(f"output_{i}" for i in range(len(interface.output_widths)))
    )
    drawing = _Diagram(show_shapes=show_shapes, show_trainable=show_trainable, rankdir=rankdir)
    adapters: list[tuple[str | None, ...]] = []
    for output in (False, True):
        labels: list[str | None] = []
        if workflow is not None:
            modules = workflow.output_scalers if output else workflow.input_scalers
            names = output_names if output else input_names
            for index, (name, module) in enumerate(zip(names, modules)):
                label = None
                if workflow._transform_active(output=output, index=index):
                    label = f"{'Inverse output' if output else 'Input'} transform: {name}\n{type(module).__name__}"
                    if hasattr(module, "method"):
                        label += f" ({module.method})"
                labels.append(label)
        adapters.append(tuple(labels))
    inputs, outputs = drawing.graph(
        graph,
        "",
        input_names,
        output_names,
        input_adapters=adapters[0],
        output_adapters=adapters[1],
    )
    if workflow is not None:
        plan = workflow._feedback_plan(validate_widths=False)
        for route in plan.routes:
            drawing.edge(
                outputs[route.output_index],
                inputs[route.input_index],
                f"feedback: delay={route.delay}; prediction lead={plan.leads[route.output_index]}",
                feedback=True,
            )
    drawing.lines.append("}")
    return "\n".join(drawing.lines)


def plot_model(
    model: nn.Module,
    *,
    save_path: str | Path | None = None,
    format: str | None = None,
    show_shapes: bool = True,
    show_trainable: bool = False,
    rankdir: str = "LR",
) -> Any:
    """Return a notebook-displayable Graphviz source, or save exactly one file.

    DOT needs no optional dependencies. SVG/PNG/PDF use `resdag[viz]` and the
    Graphviz system executable. An absent format is inferred from the suffix;
    without a suffix it defaults to SVG. No system viewer is launched.
    """
    suffix = Path(save_path).suffix.lower().lstrip(".") if save_path is not None else ""
    if format is not None and not isinstance(format, str):
        raise TypeError("format must be a string or None")
    selected = format.lower() if isinstance(format, str) else suffix or "svg"
    if selected not in DIAGRAM_FORMATS or (suffix and suffix != selected):
        raise ValueError(f"Diagram format/suffix must agree and be one of {DIAGRAM_FORMATS}")
    dot = diagram_dot(
        model, show_shapes=show_shapes, show_trainable=show_trainable, rankdir=rankdir
    )
    destination = None if save_path is None else Path(save_path)
    if destination is not None and not suffix:
        destination = destination.with_suffix(f".{selected}")
    if selected == "dot":
        if destination is None:
            return dot
        destination.write_text(dot, encoding="utf-8")
        return destination
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "Diagram rendering requires `pip install resdag[viz]`; DOT export works without it"
        ) from exc
    source = graphviz.Source(dot, format=selected)
    if destination is None:
        return source
    try:
        payload = source.pipe(format=selected)
    except graphviz.ExecutableNotFound as exc:
        raise RuntimeError(
            "Install the Graphviz system executable (e.g. brew install graphviz or apt install graphviz), or save .dot"
        ) from exc
    destination.write_bytes(payload)
    return destination
