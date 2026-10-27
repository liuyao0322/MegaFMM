"""Safe input loading and provenance for application experiments.

NPZ files contain named numerical arrays and, optionally, a scalar Unicode
``metadata_json`` value. PT files contain either a tensor dictionary or
``{"tensors": tensor_dictionary, "metadata": json_dictionary}``. Pickles that
require arbitrary Python objects are deliberately unsupported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _json_metadata(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Input metadata must be a JSON object")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Input metadata must contain only finite JSON values") from exc


def load_input_data(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load named tensors without executable pickle objects or object arrays."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Application input does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata: dict = {}
    tensors: dict[str, torch.Tensor] = {}
    if source.suffix.lower() == ".npz":
        import numpy as np

        with np.load(source, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                if name == "metadata_json":
                    if array.ndim != 0 or array.dtype.kind not in "US":
                        raise ValueError("metadata_json must be a scalar string")
                    metadata = _json_metadata(json.loads(str(array.item())))
                elif array.dtype.kind in "fiu":
                    tensors[name] = torch.from_numpy(array.copy())
                else:
                    raise ValueError(f"Input {name!r} must be a real numerical array")
        schema = "megafmm.application.npz.v1"
    elif source.suffix.lower() in (".pt", ".pth"):
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("PT application input must be a tensor dictionary")
        if "tensors" in payload:
            if set(payload) - {"tensors", "metadata"}:
                raise ValueError("Wrapped PT input supports only tensors and metadata")
            metadata = _json_metadata(payload.get("metadata", {}))
            payload = payload["tensors"]
        if not isinstance(payload, dict):
            raise ValueError("PT tensors must be a dictionary")
        for name, tensor in payload.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError("PT input keys must be strings and values must be tensors")
            if tensor.is_complex() or tensor.dtype == torch.bool or tensor.layout != torch.strided:
                raise ValueError(f"Input {name!r} must be a dense real numerical tensor")
            tensors[name] = tensor.detach().cpu().contiguous()
        schema = "megafmm.application.pt.v1"
    else:
        raise ValueError("Application input must have extension .npz, .pt, or .pth")
    if not tensors:
        raise ValueError("Application input contains no numerical tensors")
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Input {name!r} contains non-finite values")
    return tensors, {
        "scope": "imported_input",
        "path": str(source),
        "sha256": digest.hexdigest(),
        "schema": schema,
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "source_metadata": metadata,
        "task_quality_verified": False,
    }


def parse_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value in (torch.float32, torch.float16, torch.bfloat16):
            return value
        raise ValueError("Application dtype must be FP32, FP16, or BF16")
    names = {
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }
    try:
        return names[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported application dtype: {value}") from exc


def require_matrix(data: dict[str, torch.Tensor], *names: str) -> torch.Tensor:
    for name in names:
        if name in data:
            tensor = data[name]
            if tensor.ndim != 2 or min(tensor.shape) < 1:
                raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
            return tensor
    raise ValueError(f"Input requires one of these matrix keys: {', '.join(names)}")


def symmetric_matrix(matrix: torch.Tensor, name: str) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    matrix = matrix.double()
    scale = torch.linalg.vector_norm(matrix).clamp_min(torch.finfo(torch.float64).tiny)
    asymmetry = torch.linalg.vector_norm(matrix - matrix.T) / scale
    if asymmetry.item() > 1e-5:
        raise ValueError(f"{name} must be symmetric (relative asymmetry {asymmetry.item():.3g})")
    return (matrix + matrix.T) * 0.5


def random_spd(n: int, condition_number: float, generator: torch.Generator) -> torch.Tensor:
    if n < 1 or condition_number < 1 or not torch.isfinite(torch.tensor(condition_number)):
        raise ValueError("SPD size must be positive and condition_number must be finite and >= 1")
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    eigenvalues = torch.logspace(0, -torch.log10(torch.tensor(condition_number)).item(), n,
                                dtype=torch.float64)
    return (q * eigenvalues) @ q.T


def describe_tensor(value: torch.Tensor) -> dict:
    """Hash actual stored values, including their dtype/shape, outside timing."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""),
            "sha256": digest.hexdigest(), "minimum": value.min().item(),
            "maximum": value.max().item()}
