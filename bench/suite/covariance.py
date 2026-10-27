"""Dense finite-horizon covariance propagation for linear stochastic dynamics.

For ``x[t+1] = A x[t] + w[t]`` with independent process noise of covariance Q,
``P[t+1] = A P[t] A.T + Q``.  Every timed invocation starts from the same P0.
Only the two matrix products per step use the selected GEMM implementation.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .applications import Application, _mm, _relative_error


def _orthogonal_circulant(n: int, radius: float, generator: torch.Generator) -> torch.Tensor:
    """Build a real normal transition with all eigenvalue magnitudes = radius.

    A circulant's eigenvalues are its first column's Fourier transform.  Giving
    that transform unit-modulus conjugate pairs produces a real orthogonal
    matrix in exact arithmetic.  This costs O(n**2) storage/work to materialize,
    without a QR or eigendecomposition.  Random phases mix the coordinates even
    at large n; a Frobenius-normalized random matrix would instead approach a
    zero transition as n increases.
    """
    spectrum = torch.empty(n, dtype=torch.complex128)
    spectrum[0] = 1
    pairs = (n - 1) // 2
    if pairs:
        phases = 2 * math.pi * torch.rand(pairs, generator=generator, dtype=torch.float64)
        values = torch.polar(torch.ones_like(phases), phases)
        spectrum[1:pairs + 1] = values
        spectrum[-pairs:] = values.flip(0).conj()
    if n % 2 == 0:
        spectrum[n // 2] = -1
    first_column = torch.fft.ifft(spectrum).real * radius
    matrix = torch.empty(n, n, dtype=torch.float64)
    columns = torch.arange(n)
    # Avoid an additional n-by-n int64 index array in the large-square suite.
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n))[:, None]
        matrix[start:start + 256] = first_column[(rows - columns) % n]
    return matrix


class CovariancePropagation(Application):
    """FP32 covariance propagation, checked against its stored-input FP64 path."""

    kind = "covariance"

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        super().__init__(case, device, seed)
        if self.dtype != torch.float32:
            raise ValueError("covariance currently requires dtype fp32")
        if self.data:
            raise ValueError("covariance currently supports controlled synthetic inputs only")
        n = case.get("n", 512)
        if type(n) is not int or n < 1:
            raise ValueError("covariance n must be a positive integer")
        radius = float(case.get("transition_radius", 0.95))
        if not math.isfinite(radius) or not 0 < radius < 1:
            raise ValueError("transition_radius must be finite and strictly between zero and one")
        self.n = n
        transition_source = _orthogonal_circulant(n, radius, self.generator)
        stored_transition = transition_source.float()
        # For a normal exact transition, ||delta A||2 <= ||delta A||F.  The
        # diagnostic includes FP32 conversion and is conservative; no measured
        # spectral radius or exact floating-point orthogonality is claimed.
        rounding_bound = torch.linalg.vector_norm(stored_transition.double() - transition_source).item()
        self.transition = self._prepare_tensor(stored_transition)
        eye = torch.eye(n, device=self.device, dtype=torch.float64)

        def covariance(scale: float) -> torch.Tensor:
            random = torch.randn(n, n, generator=self.generator, dtype=torch.float64).to(self.device)
            source = ((random @ random.T) / n + 0.25 * eye) * scale
            return self._prepare_tensor(source)

        self.initial_covariance = covariance(1.0)
        self.process_covariance = covariance(0.05)
        self._set_thresholds({"relative_error": 1e-5, "symmetry_error": 1e-5,
                              "cholesky_failure": 0},
                             ("max_absolute_error", "trace_relative_error"))
        self.details = {
            "n": n,
            "algorithm": "finite-horizon covariance propagation: P <- (A @ P) @ A.T + Q",
            "scientific_use": "covariance prediction in a linear stochastic dynamical system",
            "scope": "controlled dense covariance prediction; no observation/Kalman correction or complete control simulation",
            "transition_generator": "real orthogonal circulant from seeded conjugate-symmetric unit Fourier phases, scaled by transition_radius",
            "transition_radius": radius,
            "transition_fp32_perturbation_frobenius": rounding_bound,
            "transition_2norm_bound_from_generator_and_fp32_rounding": radius + rounding_bound,
            "transition_bound_note": "analytic exact-arithmetic generator norm plus FP32 rounding Frobenius bound; FFT construction roundoff is not certified",
            "covariance_generator": "P0=R0@R0.T/n+0.25I; Q=0.05*(R1@R1.T/n+0.25I)",
            "input_generation": "seeded CPU FP64 RNG; covariance Gram products on selected device in FP64 outside timing; inputs stored and hashed as FP32",
            "reference": "independent PyTorch FP64 recurrence from actual stored FP32 A, P0 and Q",
            "output_rounding": "FP32 GEMM outputs and FP32 covariance state after each update",
            "gemms_per_iteration": 2,
            "gemms_per_execute": 2 * self.iterations,
            "nominal_flops": 4 * self.iterations * n ** 3,
            "flop_convention": "classical equivalent GEMM work 4*iterations*n^3; covariance additions excluded; not SW executed instructions",
            "timing_includes": "both GEMMs, output allocation, transition transpose materialization, required casts/packing and covariance addition each step",
            "positive_definiteness_check": "FP64 Cholesky of (output+output.T)/2 outside timing; symmetry is gated separately",
        }

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        covariance = self.initial_covariance
        for _ in range(self.iterations):
            product = _mm(backend, self.transition, covariance, self.dtype)
            covariance = _mm(backend, product, self.transition.T, self.dtype) + self.process_covariance
        self.last_iterations = self.iterations
        return {"result": covariance}

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            transition = self.transition.double()
            covariance = self.initial_covariance.double()
            noise = self.process_covariance.double()
            for _ in range(self.iterations):
                covariance = (transition @ covariance) @ transition.T + noise
            self._reference = {"result": covariance}
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        result, expected = output["result"].double(), reference["result"].double()
        difference = result - expected
        trace, reference_trace = torch.trace(result).item(), torch.trace(expected).item()
        finite = bool(torch.isfinite(result).all().item())
        if finite:
            _, info = torch.linalg.cholesky_ex((result + result.T) * 0.5, check_errors=False)
            cholesky_failure = int(info.item() != 0)
        else:
            cholesky_failure = 1
        error = _relative_error(result, expected)
        metrics = {
            "relative_error": error,
            "relative_frobenius_error": error,
            "absolute_frobenius_error": torch.linalg.vector_norm(difference).item(),
            "max_absolute_error": difference.abs().max().item(),
            "symmetry_error": _relative_error(result.T, result),
            "trace": trace,
            "reference_trace": reference_trace,
            "trace_relative_error": abs(trace - reference_trace) / max(abs(reference_trace), torch.finfo(torch.float64).tiny),
            "cholesky_failure": cholesky_failure,
            "iterations": self.last_iterations,
        }
        return self._finish_validation(metrics, output)
