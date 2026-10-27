"""Single-device FP32 LINPACK-style solve; this is not the FP64 HPL benchmark."""
from __future__ import annotations

import math
from typing import Any

import torch

from .applications import Application, _relative_error
from .data import require_matrix


class Linpack(Application):
    """Right-looking blocked LU with partial pivoting across the full panel.

    PyTorch factors each rectangular panel and performs triangular solves.
    The selected GEMM backend performs every trailing Schur-complement update.
    Row permutations, copies, allocations, checks and host synchronization are
    part of each execution. Only plan setup/tuning and validation are untimed.
    """

    kind = "linpack"

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        super().__init__(case, device, seed)
        if self.dtype != torch.float32:
            raise ValueError("linpack currently requires dtype fp32")
        block_size = case.get("block_size", 128)
        if type(block_size) is not int or block_size < 1:
            raise ValueError("linpack block_size must be a positive integer")
        self.block_size = block_size
        if self.data:
            matrix = require_matrix(self.data, "A")
            rhs = self.data.get("b")
            if rhs is None:
                raise ValueError("Imported linpack input requires A and b")
        else:
            n = case.get("n", 512)
            if type(n) is not int or n < 1:
                raise ValueError("linpack n must be a positive integer")
            matrix = 2 * torch.rand(n, n, generator=self.generator, dtype=torch.float64) - 1
            # Form b from the stored matrix, then round it independently to FP32.
            # The FP64 solve of the stored A/b is the authoritative reference.
            matrix = matrix.float()
            rhs = matrix.double() @ torch.ones(n, 1, dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise ValueError("linpack A must be a nonempty square matrix")
        self.n = matrix.shape[0]
        if "n" in case and case["n"] != self.n:
            raise ValueError("linpack n must match the imported A dimension")
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or tuple(rhs.shape) != (self.n, 1):
            raise ValueError("linpack b must have shape (n,) or (n, 1)")
        self.matrix = self._prepare_tensor(matrix)
        self.rhs = self._prepare_tensor(rhs)
        self.iterations = self.last_iterations = math.ceil(self.n / self.block_size)
        self._set_thresholds({"scaled_residual": 16.0},
                             ("relative_error", "backward_error", "residual_inf"))
        update_shapes = [[self.n - end, self.n - end, self.block_size]
                         for end in range(self.block_size, self.n, self.block_size)]
        self.details = {
            "benchmark": "linpack_style_fp32", "standard_hpl": False,
            "n": self.n, "block_size": self.block_size, "nrhs": 1,
            "distribution": "imported" if self.data else "uniform",
            "rhs_generation": "imported" if self.data else "stored A times ones in FP64, then rounded to FP32",
            "factorization": "blocked right-looking LU with full-panel partial pivoting",
            "panel_and_triangular_solver": "torch FP32 (identical across GEMM backends)",
            "gemms_per_execute": len(update_shapes), "gemm_shapes_mnk": update_shapes,
            "nominal_flops": (2.0 / 3.0) * self.n ** 3 + 2.0 * self.n ** 2,
            "flop_convention": "(2/3)*n^3 + 2*n^2; equivalent classical solve work, not SW executed instructions",
            "residual_formula": "||Ax-b||inf / (n * finfo(fp32).eps * (||A||inf*||x||inf + ||b||inf))",
            "residual_epsilon": torch.finfo(torch.float32).eps,
            "reference": "independent FP64 solve of actual stored FP32 A and b",
            "timing_basis": "wall", "supports_cuda_graph": False,
            "timing_includes": "input clones, panel LU, pivot/status host sync, row swaps, triangular solves, casts/packing, GEMM updates and final solve",
        }

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        lu, rhs = self.matrix.clone(), self.rhs.clone()
        permutation = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.block_size):
            end = min(start + self.block_size, self.n)
            panel, pivots, info = torch.linalg.lu_factor_ex(lu[start:, start:end], check_errors=False)
            if info.item() != 0:
                raise RuntimeError(f"linpack singular panel at column {start}: LU info={info.item()}")
            # LAPACK pivots are one-based successive swaps, not a permutation.
            order = list(range(self.n - start))
            for row, pivot in enumerate(pivots.cpu().tolist()):
                other = pivot - 1
                order[row], order[other] = order[other], order[row]
            order = torch.tensor(order, device=self.device, dtype=torch.long)
            # Include already factored columns so previous L multipliers follow
            # the row swaps; the freshly factored panel is already permuted.
            lu[start:] = lu[start:].index_select(0, order)
            rhs[start:] = rhs[start:].index_select(0, order)
            permutation[start:] = permutation[start:].index_select(0, order)
            lu[start:, start:end] = panel
            if end < self.n:
                upper = torch.linalg.solve_triangular(
                    lu[start:end, start:end], lu[start:end, end:],
                    upper=False, unitriangular=True)
                lu[start:end, end:] = upper
                product = backend.gemm(lu[end:, start:end].contiguous(), upper.contiguous())
                lu[end:, end:] -= product
        y = torch.linalg.solve_triangular(lu, rhs, upper=False, unitriangular=True)
        x = torch.linalg.solve_triangular(lu, y, upper=True)
        return {"result": x, "lu": lu, "permutation": permutation}

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            self._reference = {"result": torch.linalg.solve(self.matrix.double(), self.rhs.double())}
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        a, b, x = self.matrix.double(), self.rhs.double(), output["result"].double()
        residual = (a @ x - b).abs().max().item()
        denominator = (a.abs().sum(dim=1).max() * x.abs().max() + b.abs().max()).item()
        backward_error = residual / denominator if denominator > 0 else (0.0 if residual == 0 else float("inf"))
        metrics = {"residual_inf": residual, "backward_error": backward_error,
                   "scaled_residual": backward_error / (self.n * torch.finfo(torch.float32).eps),
                   "scaled_residual_fp64_diagnostic": backward_error / (self.n * torch.finfo(torch.float64).eps),
                   "relative_error": _relative_error(x, reference["result"]),
                   "residual_epsilon": torch.finfo(torch.float32).eps}
        return self._finish_validation(metrics, output)
