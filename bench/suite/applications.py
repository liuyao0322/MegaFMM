"""Application-level experiments with explicit numerical quality contracts.

The GEMM backend has one method, ``gemm(a, b)``, returning FP32. Each algorithm
explicitly rounds that output at its declared storage boundary. Reference
calculations use independent FP64 PyTorch operations outside measured regions.
Synthetic inputs establish algorithm behavior, never model/physical task quality.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from .data import describe_tensor, load_input_data, parse_dtype, random_spd, require_matrix, symmetric_matrix


class _DoubleBackend:
    def gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.double() @ b.double()


def _mm(backend: Any, a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Contiguity and casts belong to the measured application, including transpose
    # materialization. Inputs are never overwritten by an application invocation.
    return backend.gemm(a.contiguous(), b.contiguous()).to(dtype=dtype)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.double() - expected.double())
    denominator = torch.linalg.vector_norm(expected.double())
    return (numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)).item()


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.double().flatten(), expected.double().flatten()
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if an.item() == 0 or bn.item() == 0:
        return 1.0 if an.item() == bn.item() else 0.0
    return ((a @ b) / (an * bn)).item()


def _finite(output: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in output.values())


@contextmanager
def _strict_fp32_checker():
    """Keep stop decisions independent of a backend's TF32 setting."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


class Application:
    kind = "abstract"
    supports_time_to_quality = False

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        self.case = copy.deepcopy(case)
        self.device = torch.device(device)
        self.dtype = parse_dtype(case.get("dtype", "fp32"))
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.iterations = int(case.get("iterations", 10))
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        self.last_iterations = self.iterations
        self._reference: dict[str, torch.Tensor] | None = None
        self.quality_thresholds: dict[str, float] = {}
        self.details: dict = {}
        self.input_records: list[dict] = []
        if case.get("input_path"):
            self.data, self.provenance = load_input_data(case["input_path"])
        else:
            self.data = {}
            self.provenance = {"scope": "synthetic_algorithm", "seed": self.seed,
                               "task_quality_verified": False}

    def _prepare_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = value.detach().to(device="cpu", dtype=self.dtype).contiguous()
        if not torch.isfinite(tensor).all().item():
            raise ValueError("Input conversion overflowed the selected storage dtype")
        self.input_records.append(describe_tensor(tensor))
        return tensor.to(device=self.device)

    def _random_matrix(self, rows: int, columns: int) -> torch.Tensor:
        distribution = self.case.get("distribution", "normal")
        if distribution == "normal":
            return torch.randn(rows, columns, dtype=torch.float64, generator=self.generator)
        if distribution == "uniform":
            return 2 * torch.rand(rows, columns, dtype=torch.float64, generator=self.generator) - 1
        raise ValueError("distribution must be normal or uniform")

    def _scale_input(self, value: torch.Tensor) -> torch.Tensor:
        scale = float(self.case.get("input_scale", 1))
        dynamic_range = float(self.case.get("dynamic_range", 1))
        if not math.isfinite(scale) or not math.isfinite(dynamic_range) or dynamic_range < 1:
            raise ValueError("input_scale must be finite and dynamic_range must be finite and >= 1")
        scales = torch.logspace(-0.5 * math.log10(dynamic_range), 0.5 * math.log10(dynamic_range),
                                value.shape[1], dtype=torch.float64)
        return value.double() * scales * scale

    def _set_thresholds(self, defaults: dict[str, float], optional_metrics: tuple[str, ...] = ()) -> None:
        overrides = self.case.get("quality", self.case.get("quality_thresholds", {}))
        if not isinstance(overrides, dict):
            raise ValueError("quality must be a metric-to-upper-bound dictionary")
        unknown = set(overrides) - set(defaults) - set(optional_metrics)
        if unknown:
            raise ValueError(f"Unknown {self.kind} quality metrics: {sorted(unknown)}")
        self.quality_thresholds = {**defaults, **overrides}
        for name, threshold in self.quality_thresholds.items():
            if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Quality threshold {name} must be finite and nonnegative")

    def _finish_validation(self, metrics: dict, output: dict[str, torch.Tensor]) -> dict:
        metrics["finite"] = _finite(output)
        metrics["quality_thresholds"] = dict(self.quality_thresholds)
        metrics["quality_pass"] = metrics["finite"] and all(
            math.isfinite(float(metrics[key])) and float(metrics[key]) <= bound
            for key, bound in self.quality_thresholds.items())
        if "converged" in output:
            metrics["quality_pass"] = metrics["quality_pass"] and bool(output["converged"].item())
        return metrics

    def passes_quality(self, metrics: dict) -> bool:
        return bool(metrics.get("quality_pass", False))

    def metadata(self) -> dict:
        return {"application": self.kind, "dtype": str(self.dtype).replace("torch.", ""),
                "iterations": self.iterations, "seed": self.seed,
                "provenance": copy.deepcopy(self.provenance),
                "quality_thresholds": dict(self.quality_thresholds),
                "stored_inputs": copy.deepcopy(self.input_records),
                "input_scale": self.case.get("input_scale", 1),
                "dynamic_range": self.case.get("dynamic_range", 1),
                "distribution": self.case.get("distribution", "normal"),
                "supports_time_to_quality": self.supports_time_to_quality,
                **copy.deepcopy(self.details)}

    def _dynamic_settings(self) -> tuple[int, int, float]:
        maximum = int(self.case.get("max_iterations", self.iterations))
        check_every = int(self.case.get("check_every", 1))
        tolerance = float(self.case.get("stop_tolerance", self.quality_thresholds.get("residual", 1e-3)))
        if maximum < 1 or check_every < 1 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("max_iterations/check_every must be positive; stop_tolerance must be finite and > 0")
        return maximum, check_every, tolerance


class MatrixFunction(Application):
    kind = "matrix_function"
    supports_time_to_quality = True

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        super().__init__(case, device, seed)
        self.function = case.get("function", "sqrt")
        if self.function not in ("sqrt", "inverse_sqrt"):
            raise ValueError("matrix_function function must be sqrt or inverse_sqrt")
        if self.data:
            source = symmetric_matrix(require_matrix(self.data, "S", "A", "Sigma"), "S/A/Sigma")
        else:
            source = random_spd(int(case.get("n", 64)), float(case.get("condition_number", 100)), self.generator)
        regularization = float(case.get("regularization", 0))
        if regularization < 0 or not math.isfinite(regularization):
            raise ValueError("regularization must be finite and nonnegative")
        source = source + regularization * torch.eye(source.shape[0], dtype=torch.float64)
        self.original_matrix = source.to(device=self.device, dtype=torch.float64)
        self.matrix = self._prepare_tensor(source)
        self.n = self.matrix.shape[0]
        values = torch.linalg.eigvalsh(self.matrix.double())
        if values[0].item() <= 0:
            raise ValueError("Matrix function input must remain positive definite after storage conversion")
        self.eye = torch.eye(self.n, device=self.device, dtype=self.dtype)
        self._set_thresholds({"residual": 5e-3, "relative_error": 5e-3, "symmetry_error": 5e-4})
        self.details = {"function": self.function, "n": self.n, "gemms_per_iteration": 3,
                        "regularization": regularization, "condition_number": (values[-1] / values[0]).item(),
                        "output_rounding": "each GEMM rounded to application dtype",
                        "dynamic_checker": "independent torch FP32 matrix products and host convergence sync, included in timing"}

    def _initial(self, dtype: torch.dtype):
        matrix = self.matrix.to(dtype=dtype)
        eye = self.eye.to(dtype=dtype)
        # FP32 reduction avoids a BF16/FP16 trace overflow; FP64 reference keeps FP64.
        alpha = torch.trace(matrix if dtype == torch.float64 else matrix.float())
        return matrix / alpha, eye, alpha, eye

    def _step(self, backend: Any, y: torch.Tensor, z: torch.Tensor,
              eye: torch.Tensor, dtype: torch.dtype):
        t = (3 * eye - _mm(backend, z, y, dtype)) * 0.5
        return _mm(backend, y, t, dtype), _mm(backend, t, z, dtype)

    def _result(self, y: torch.Tensor, z: torch.Tensor, alpha: torch.Tensor):
        result = y * torch.sqrt(alpha) if self.function == "sqrt" else z / torch.sqrt(alpha)
        return {"result": result}

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        y, z, alpha, eye = self._initial(self.dtype)
        for _ in range(self.iterations):
            y, z = self._step(backend, y, z, eye, self.dtype)
        self.last_iterations = self.iterations
        return self._result(y, z, alpha)

    def _residual(self, result: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        result = result.to(dtype)
        matrix = self.original_matrix.to(dtype)
        if self.function == "sqrt":
            return torch.linalg.vector_norm(result @ result - matrix) / torch.linalg.vector_norm(matrix)
        eye = torch.eye(self.n, device=self.device, dtype=dtype)
        return torch.linalg.vector_norm(result @ matrix @ result - eye) / math.sqrt(self.n)

    @torch.no_grad()
    def execute_to_quality(self, backend: Any) -> dict[str, torch.Tensor]:
        maximum, check_every, tolerance = self._dynamic_settings()
        y, z, alpha, eye = self._initial(self.dtype)
        converged = False
        for iteration in range(1, maximum + 1):
            y, z = self._step(backend, y, z, eye, self.dtype)
            if iteration % check_every == 0 or iteration == maximum:
                with _strict_fp32_checker():
                    residual = self._residual(self._result(y, z, alpha)["result"], torch.float32).item()
                if math.isfinite(residual) and residual <= tolerance:
                    converged = True
                    break
        self.last_iterations = iteration
        result = self._result(y, z, alpha)
        result["iterations"] = torch.tensor(iteration, device=self.device)
        result["converged"] = torch.tensor(converged, device=self.device)
        return result

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            values, vectors = torch.linalg.eigh(self.original_matrix)
            factors = torch.sqrt(values) if self.function == "sqrt" else torch.rsqrt(values)
            exact = (vectors * factors) @ vectors.T
            y, z, alpha, eye = self._initial(torch.float64)
            for _ in range(self.iterations):
                y, z = self._step(_DoubleBackend(), y, z, eye, torch.float64)
            self._reference = {"result": exact, "algorithm_result": self._result(y, z, alpha)["result"]}
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        result = output["result"].double()
        metrics = {"residual": self._residual(result, torch.float64).item(),
                   "relative_error": _relative_error(result, reference["result"]),
                   "symmetry_error": _relative_error(result.T, result),
                   "iterations": int(output["iterations"].item()) if "iterations" in output else self.last_iterations}
        if "algorithm_result" in reference:
            metrics["algorithm_reference_error"] = _relative_error(result, reference["algorithm_result"])
        if "converged" in output:
            metrics["converged"] = output["converged"].item()
        return self._finish_validation(metrics, output)


class Muon(Application):
    kind = "muon"

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        case = {"iterations": 5, **case}
        super().__init__(case, device, seed)
        if self.data:
            gradient = require_matrix(self.data, "G", "momentum", "gradient")
        else:
            m, n = int(case.get("m", case.get("n", 64))), int(case.get("n", 64))
            if min(m, n) < 1:
                raise ValueError("Muon dimensions must be positive")
            gradient = self._random_matrix(m, n)
        self.original_gradient = self._scale_input(gradient).to(device=self.device)
        self.gradient = self._prepare_tensor(self.original_gradient)
        self.transpose = self.gradient.shape[0] > self.gradient.shape[1]
        self.coefficients = (3.4445, -4.775, 2.0315)
        self._set_thresholds({"relative_error": 0.1, "direction_error": 0.01}, ("source_relative_error",))
        self.details = {"shape": list(self.gradient.shape), "gemms_per_iteration": 3,
                        "coefficients": list(self.coefficients), "transpose_for_gram": self.transpose,
                        "reference_contract": "FP64 same finite-step polynomial on stored input; SVD polar is diagnostic only",
                        "source_reference_contract": "source_relative_error includes input storage rounding; input_quantization_reference_error isolates its FP64 same-polynomial effect",
                        "output_rounding": "each GEMM and elementwise operation uses application dtype; native FP32 GEMM outputs explicitly cast",
                        "scope": "optimizer_update_replay; training quality requires separate training experiment"}

    def _run(self, backend: Any, dtype: torch.dtype, source: bool = False) -> dict[str, torch.Tensor]:
        x = (self.original_gradient if source else self.gradient).to(dtype=dtype)
        if self.transpose:
            x = x.T
        x = x / (x.norm() + 1e-7)
        a, b, c = self.coefficients
        for _ in range(self.iterations):
            gram = _mm(backend, x, x.T, dtype)
            polynomial = b * gram + c * _mm(backend, gram, gram, dtype)
            x = a * x + _mm(backend, polynomial, x, dtype)
        return {"result": x.T.contiguous() if self.transpose else x}

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        return self._run(backend, self.dtype)

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            reference = self._run(_DoubleBackend(), torch.float64)
            reference["source_result"] = self._run(_DoubleBackend(), torch.float64, source=True)["result"]
            u, _, vh = torch.linalg.svd(self.gradient.double(), full_matrices=False)
            reference["polar"] = u @ vh
            self._reference = reference
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        result = output["result"].double()
        expected = reference["result"]
        gram = result @ result.T if result.shape[0] <= result.shape[1] else result.T @ result
        eye = torch.eye(gram.shape[0], device=self.device, dtype=torch.float64)
        metrics = {"relative_error": _relative_error(result, expected),
                   "direction_error": max(0.0, 1 - _cosine(result, expected)),
                   "direction_cosine": _cosine(result, expected),
                   "norm_ratio": (torch.linalg.vector_norm(result) / torch.linalg.vector_norm(expected).clamp_min(1e-300)).item(),
                   "orthogonality_residual": (torch.linalg.vector_norm(gram - eye) / math.sqrt(gram.shape[0])).item(),
                   "iterations": self.iterations}
        if "polar" in reference:
            metrics["polar_relative_error_diagnostic"] = _relative_error(result, reference["polar"])
        if "source_result" in reference:
            metrics["source_relative_error"] = _relative_error(result, reference["source_result"])
            metrics["input_quantization_reference_error"] = _relative_error(expected, reference["source_result"])
        return self._finish_validation(metrics, output)


class MLP(Application):
    kind = "mlp"

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        super().__init__(case, device, seed)
        source_metadata = self.provenance.get("source_metadata", {})
        self.activation = case.get("activation", source_metadata.get("mlp_kind", "gelu"))
        if self.activation not in ("gelu", "swiglu"):
            raise ValueError("MLP activation must be gelu or swiglu")
        self.gelu_approximate = case.get("gelu_approximate", source_metadata.get("gelu_approximate", "none"))
        if self.gelu_approximate not in ("none", "tanh"):
            raise ValueError("gelu_approximate must be none or tanh")
        if self.data:
            x = require_matrix(self.data, "X")
            w1, w2 = require_matrix(self.data, "W1"), require_matrix(self.data, "W2")
            w3 = require_matrix(self.data, "W3") if self.activation == "swiglu" else None
        else:
            tokens, hidden, intermediate = (int(case.get(key, default)) for key, default in
                                            (("tokens", 64), ("hidden", 64), ("intermediate", 256)))
            if min(tokens, hidden, intermediate) < 1:
                raise ValueError("MLP dimensions must be positive")
            x = self._random_matrix(tokens, hidden)
            w1 = torch.randn(hidden, intermediate, dtype=torch.float64, generator=self.generator) / math.sqrt(hidden)
            w2 = torch.randn(intermediate, hidden, dtype=torch.float64, generator=self.generator) / math.sqrt(intermediate)
            w3 = torch.randn(hidden, intermediate, dtype=torch.float64, generator=self.generator) / math.sqrt(hidden) if self.activation == "swiglu" else None
        if x.shape[1] != w1.shape[0] or w1.shape[1] != w2.shape[0]:
            raise ValueError("MLP requires X[T,D], W1[D,H], W2[H,O] mathematical layouts")
        if w3 is not None and w3.shape != w1.shape:
            raise ValueError("SwiGLU W3 must have the same shape as W1")
        self.original_x, self.original_w1, self.original_w2 = (
            tensor.to(device=self.device) for tensor in (self._scale_input(x), w1, w2))
        self.x, self.w1, self.w2 = map(self._prepare_tensor, (self.original_x, self.original_w1, self.original_w2))
        self.original_w3 = w3.to(device=self.device) if w3 is not None else None
        self.w3 = self._prepare_tensor(w3) if w3 is not None else None
        self.biases = []
        self.original_biases = []
        for key, length in (("b1", w1.shape[1]), ("b2", w2.shape[1]), ("b3", w1.shape[1])):
            if self.data:
                value = self.data.get(key, torch.zeros(length, dtype=torch.float64))
            elif case.get("bias", True):
                value = torch.randn(length, dtype=torch.float64, generator=self.generator) * 0.01
            else:
                value = torch.zeros(length, dtype=torch.float64)
            if value.shape != (length,):
                raise ValueError(f"MLP {key} must have shape [{length}]")
            self.original_biases.append(value.to(device=self.device))
            self.biases.append(self._prepare_tensor(value))
        self._set_thresholds({"relative_error": 0.05, "direction_error": 0.01}, ("source_relative_error",))
        self.details = {"activation": self.activation, "gelu_approximate": self.gelu_approximate,
                        "input_shape": list(x.shape),
                        "weight_shapes": [list(w1.shape), list(w2.shape)] + ([list(w3.shape)] if w3 is not None else []),
                        "gemms_per_execute": 3 if self.activation == "swiglu" else 2,
                        "layout": "mathematical X[T,D], W1[D,H], W2[H,O], W3[D,H]",
                        "reference_contract": "relative_error compares FP64 module on stored inputs; source_relative_error also includes input storage rounding",
                        "output_rounding": "each GEMM rounded to application dtype before bias and activation",
                        "scope": "MLP module replay; model perplexity requires separate model experiment"}

    def _run(self, backend: Any, dtype: torch.dtype, source: bool = False) -> dict[str, torch.Tensor]:
        tensors = (self.original_x, self.original_w1, self.original_w2) if source else (self.x, self.w1, self.w2)
        x, w1, w2 = (tensor.to(dtype=dtype) for tensor in tensors)
        b1, b2, b3 = (tensor.to(dtype=dtype) for tensor in (self.original_biases if source else self.biases))
        first = _mm(backend, x, w1, dtype) + b1
        if self.activation == "gelu":
            hidden = F.gelu(first, approximate=self.gelu_approximate)
        else:
            w3 = self.original_w3 if source else self.w3
            gate = _mm(backend, x, w3.to(dtype=dtype), dtype) + b3
            hidden = F.silu(first) * gate
        return {"result": _mm(backend, hidden, w2, dtype) + b2}

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        return self._run(backend, self.dtype)

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            self._reference = self._run(_DoubleBackend(), torch.float64)
            self._reference["source_result"] = self._run(_DoubleBackend(), torch.float64, source=True)["result"]
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        result, expected = output["result"], reference["result"]
        metrics = {"relative_error": _relative_error(result, expected),
                   "direction_error": max(0.0, 1 - _cosine(result, expected)),
                   "direction_cosine": _cosine(result, expected),
                   "max_absolute_error": (result.double() - expected.double()).abs().max().item()}
        if "source_result" in reference:
            metrics["source_relative_error"] = _relative_error(result, reference["source_result"])
            metrics["input_quantization_reference_error"] = _relative_error(expected, reference["source_result"])
        return self._finish_validation(metrics, output)


class Purification(Application):
    kind = "purification"
    supports_time_to_quality = True

    def __init__(self, case: dict, device: str | torch.device, seed: int):
        case = {"iterations": 25, **case}
        super().__init__(case, device, seed)
        preprocessing = "none; orthonormal basis"
        source_metadata = self.provenance.get("source_metadata", {})
        if self.data:
            h = symmetric_matrix(require_matrix(self.data, "H", "F"), "H/F")
            if "S" in self.data and source_metadata.get("H_basis") != "orthonormal":
                s = symmetric_matrix(self.data["S"], "S")
                if s.shape != h.shape:
                    raise ValueError("S and H/F must have identical shapes")
                values, vectors = torch.linalg.eigh(s)
                if values[0].item() <= 0:
                    raise ValueError("Nonorthogonal overlap S must be positive definite")
                inverse_sqrt = (vectors * torch.rsqrt(values)) @ vectors.T
                h = inverse_sqrt @ h @ inverse_sqrt
                preprocessing = "FP64 Lowdin Horth=S^(-1/2) H S^(-1/2), prepared outside timing"
        else:
            n = int(case.get("n", 64))
            occupied = int(case.get("occupied", case.get("nocc", n // 2)))
            gap = float(case.get("gap", 0.2))
            if n < 2 or not 0 < occupied < n or not math.isfinite(gap) or not 0 < gap < 2:
                raise ValueError("Synthetic purification requires n>=2, 0<occupied<n, 0<gap<2")
            q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=self.generator))
            eigenvalues = torch.cat((torch.linspace(-1, -gap / 2, occupied, dtype=torch.float64),
                                     torch.linspace(gap / 2, 1, n - occupied, dtype=torch.float64)))
            h = (q * eigenvalues) @ q.T
        self.original_h = h.to(device=self.device, dtype=torch.float64)
        self.h = self._prepare_tensor(h)
        self.n = self.h.shape[0]
        source_occupied = source_metadata.get("occupied", source_metadata.get("nocc"))
        occupied_setting = case.get("occupied", case.get("nocc", source_occupied))
        if occupied_setting is None:
            if self.data:
                raise ValueError("Imported purification requires case occupied/nocc or metadata occupied/nocc")
            occupied_setting = self.n // 2
        self.occupied = int(occupied_setting)
        if not 0 < self.occupied < self.n:
            raise ValueError("Purification requires 0 < occupied < n (spinless projector)")
        eigenvalues = torch.linalg.eigvalsh(self.h.double())
        self.minimum, self.maximum = eigenvalues[0].item(), eigenvalues[-1].item()
        if self.maximum <= self.minimum:
            raise ValueError("Hamiltonian must have a nonzero spectral width")
        self.eye = torch.eye(self.n, device=self.device, dtype=self.dtype)
        self._set_thresholds({"residual": 5e-4, "trace_error": 1e-3, "relative_error": 5e-3})
        self.details = {"n": self.n, "occupied": self.occupied, "gemms_per_iteration": 1,
                        "algorithm": "trace-correcting SP2: X^2 if trace(X)>occupied else 2X-X^2",
                        "preprocessing": preprocessing,
                        "scope": "frozen Hamiltonian projector; SCF convergence/energy/forces require separate SCF experiment",
                        "spectral_bounds": [self.minimum, self.maximum],
                        "occupied_gap": (eigenvalues[self.occupied] - eigenvalues[self.occupied - 1]).item(),
                        "dynamic_checker": "independent torch FP32 idempotency and trace checks with host sync, included in timing"}

    def _initial(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.maximum * self.eye.to(dtype=dtype) - self.h.to(dtype=dtype)) / (self.maximum - self.minimum)

    def _step(self, backend: Any, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        square = _mm(backend, x, x, dtype)
        trace = torch.trace(x if dtype == torch.float64 else x.float())
        return torch.where(trace > self.occupied, square, 2 * x - square)

    @torch.no_grad()
    def execute(self, backend: Any) -> dict[str, torch.Tensor]:
        x = self._initial(self.dtype)
        for _ in range(self.iterations):
            x = self._step(backend, x, self.dtype)
        self.last_iterations = self.iterations
        return {"result": x}

    @torch.no_grad()
    def execute_to_quality(self, backend: Any) -> dict[str, torch.Tensor]:
        maximum, check_every, tolerance = self._dynamic_settings()
        trace_tolerance = float(self.case.get("trace_stop_tolerance", self.quality_thresholds["trace_error"]))
        if not math.isfinite(trace_tolerance) or trace_tolerance <= 0:
            raise ValueError("trace_stop_tolerance must be finite and positive")
        x = self._initial(self.dtype)
        converged = False
        for iteration in range(1, maximum + 1):
            x = self._step(backend, x, self.dtype)
            if iteration % check_every == 0 or iteration == maximum:
                with _strict_fp32_checker():
                    checked = x.float()
                    residual = (torch.linalg.vector_norm(checked @ checked - checked) / math.sqrt(self.occupied)).item()
                    trace_error = abs(torch.trace(checked).item() - self.occupied)
                if math.isfinite(residual) and residual <= tolerance and trace_error <= trace_tolerance:
                    converged = True
                    break
        self.last_iterations = iteration
        return {"result": x, "iterations": torch.tensor(iteration, device=self.device),
                "converged": torch.tensor(converged, device=self.device)}

    @torch.no_grad()
    def make_reference(self) -> dict[str, torch.Tensor]:
        if self._reference is None:
            _, vectors = torch.linalg.eigh(self.original_h)
            occupied_vectors = vectors[:, :self.occupied]
            x = self._initial(torch.float64)
            for _ in range(self.iterations):
                x = self._step(_DoubleBackend(), x, torch.float64)
            self._reference = {"result": occupied_vectors @ occupied_vectors.T,
                               "algorithm_result": x}
        return self._reference

    @torch.no_grad()
    def validate(self, output: dict[str, torch.Tensor], reference: dict | None = None) -> dict:
        reference = self.make_reference() if reference is None else reference
        result = output["result"].double()
        expected = reference["result"]
        energy, reference_energy = torch.trace(self.original_h @ result), torch.trace(self.original_h @ expected)
        metrics = {"residual": (torch.linalg.vector_norm(result @ result - result) / math.sqrt(self.occupied)).item(),
                   "trace_error": abs(torch.trace(result).item() - self.occupied),
                   "relative_error": _relative_error(result, expected),
                   "symmetry_error": _relative_error(result.T, result),
                   "band_energy": energy.item(), "reference_band_energy": reference_energy.item(),
                   "band_energy_absolute_error": abs((energy - reference_energy).item()),
                   "iterations": int(output["iterations"].item()) if "iterations" in output else self.last_iterations}
        if "algorithm_result" in reference:
            metrics["algorithm_reference_error"] = _relative_error(result, reference["algorithm_result"])
        if "converged" in output:
            metrics["converged"] = output["converged"].item()
        return self._finish_validation(metrics, output)


def create_application(case: dict, device: str | torch.device = "cpu", seed: int = 0) -> Application:
    """Construct an application; input preparation and reference work are untimed."""
    from .linpack import Linpack
    from .covariance import CovariancePropagation

    kind = case.get("app", case.get("application", case.get("kind", case.get("name"))))
    implementations = {"matrix_function": MatrixFunction, "muon": Muon,
                       "mlp": MLP, "purification": Purification, "linpack": Linpack,
                       "covariance": CovariancePropagation}
    try:
        implementation = implementations[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown application {kind!r}; choose {', '.join(implementations)}") from exc
    return implementation(case, device, seed)
