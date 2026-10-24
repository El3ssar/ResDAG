"""Canonical compressed recurrence stored as ordinary Torch index/value buffers.

The CSR tensor is a zero-copy operator view, not another set of weights. Keeping
its three components as buffers makes ordinary module copying, placement and
restricted checkpoints work without sparse-storage deepcopy support.
"""

from typing import Any

import torch
from torch import nn

from ._structured_recurrent import one_entry_per_row_columns


def _scale_recurrent_rows(matrix: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    if matrix.layout == torch.sparse_csr:
        result = matrix.clone()
        repeats = matrix.crow_indices().diff()
        result.values().mul_(torch.repeat_interleave(gain, repeats))
        return result
    return matrix * gain[:, None]


class CompactRecurrentMixin(nn.Module):
    reservoir_size: int

    def _initialize_compact_buffers(self) -> None:
        for name in ("_hh_crow_indices", "_hh_col_indices", "_hh_values"):
            self.register_buffer(name, None)
        self.register_buffer("_hh_single_columns", None, persistent=False)
        self._hh_structure_key: tuple[int, int] | None = None

    def _structure_key(self) -> tuple[int, int] | None:
        crow = self._buffers.get("_hh_crow_indices")
        matrix = self._buffers.get("_weight_hh_csr")
        if crow is None or matrix is None:
            return None
        try:
            return crow._version, matrix._version
        except RuntimeError:
            # Inference tensors do not expose mutation counters. Keep their
            # general CSR path correct without per-step scans or rebuilding.
            return None

    def _refresh_structured_weight(self) -> None:
        key = self._structure_key()
        self._hh_structure_key = key
        self._buffers["_hh_single_columns"] = (
            one_entry_per_row_columns(self.weight_hh) if key is not None else None
        )

    @property
    def weight_hh(self) -> torch.Tensor:
        """Dense Parameter or zero-copy CSR view of frozen recurrent weights.

        Use `weight_hh.to_dense()` when a dense inspection copy is wanted.
        Sparse values can be edited in place through `weight_hh.values()`.
        """
        parameter = self._parameters.get("weight_hh")
        if parameter is not None:
            return parameter
        values = self._buffers.get("_hh_values")
        if values is None:
            raise AttributeError("Recurrent weights have not been initialized")
        view = self._buffers.get("_weight_hh_csr")
        if view is None:
            crow = self._buffers["_hh_crow_indices"]
            columns = self._buffers["_hh_col_indices"]
            assert crow is not None and columns is not None
            view = torch.sparse_csr_tensor(
                crow,
                columns,
                values,
                size=(self.reservoir_size, self.reservoir_size),
                check_invariants=True,
            )
            self._buffers["_weight_hh_csr"] = view
        return view

    @weight_hh.setter
    def weight_hh(self, value: nn.Parameter) -> None:
        self.register_parameter("weight_hh", value)

    def _store_compact_weight(self, matrix: torch.Tensor) -> None:
        matrix = matrix.detach()
        if matrix.layout != torch.sparse_csr:
            matrix = matrix.to_sparse_csr()
        self._buffers["_hh_crow_indices"] = matrix.crow_indices().clone()
        self._buffers["_hh_col_indices"] = matrix.col_indices().clone()
        self._buffers["_hh_values"] = matrix.values().clone()
        self._parameters["weight_hh"] = None
        self._buffers["_weight_hh_csr"] = None
        # Reconstruct once with invariant checks before accepting caller-owned
        # CSR data. The returned view shares only our owned cloned components.
        self.weight_hh
        self._refresh_structured_weight()

    def _store_dense_weight(self, matrix: torch.Tensor, *, trainable: bool = False) -> None:
        if matrix.layout != torch.strided:
            # MPS cannot materialize a sparse tensor; assemble on CPU first.
            if matrix.device.type == "mps":
                matrix = matrix.cpu().to_dense().to("mps")
            else:
                matrix = matrix.to_dense()
        self._parameters["weight_hh"] = nn.Parameter(matrix.detach(), requires_grad=trainable)
        for name in (
            "_hh_crow_indices",
            "_hh_col_indices",
            "_hh_values",
            "_weight_hh_csr",
            "_hh_single_columns",
        ):
            self._buffers[name] = None
        self._hh_structure_key = None

    def _prepare_compact_load(self, state_dict: dict[str, Any], prefix: str) -> None:
        names = ("_hh_crow_indices", "_hh_col_indices", "_hh_values")
        present = [prefix + name in state_dict for name in names]
        current = self._parameters.get("weight_hh")
        force_dense = not getattr(self, "_sparse_requested", False) or (
            current is not None and current.requires_grad
        )
        if all(present) and force_dense:
            shape = (self.reservoir_size, self.reservoir_size)
            sparse = torch.sparse_csr_tensor(
                state_dict[prefix + names[0]],
                state_dict[prefix + names[1]],
                state_dict[prefix + names[2]],
                size=shape,
                check_invariants=True,
            )
            state_dict[prefix + "weight_hh"] = sparse.to_dense()
            for name in names:
                state_dict.pop(prefix + name)
            present = [False] * len(names)
        if all(present):
            reference = self._parameters.get("weight_feedback")
            assert reference is not None
            self._parameters["weight_hh"] = None
            for name in names:
                tensor = state_dict[prefix + name]
                dtype = reference.dtype if name == "_hh_values" else tensor.dtype
                self._buffers[name] = torch.empty(
                    tensor.shape, dtype=dtype, device=reference.device
                )
        elif prefix + "weight_hh" in state_dict:
            reference = self._parameters.get("weight_feedback")
            assert reference is not None
            tensor = state_dict[prefix + "weight_hh"]
            current = self._parameters.get("weight_hh")
            if current is None:
                self._parameters["weight_hh"] = nn.Parameter(
                    torch.empty(tensor.shape, dtype=reference.dtype, device=reference.device),
                    requires_grad=getattr(self, "_trainable", False),
                )
            for name in names:
                self._buffers[name] = None
        self._buffers["_weight_hh_csr"] = None
        self._buffers["_hh_single_columns"] = None
        self._hh_structure_key = None
