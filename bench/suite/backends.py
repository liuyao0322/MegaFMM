"""Matched GEMM backends and explicit setup for application benchmarks.

``cublaslt`` and ``megafmm`` share the public native Plan API and FP32 output
contract.  ``torch`` executes the framework's actual matmul; its low precision
output is rounded before conversion to FP32, which is reported in metadata.
PyTorch is imported only when a Backend is instantiated.
"""
from __future__ import annotations

import ctypes
from contextlib import nullcontext
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


class BackendError(RuntimeError):
    """Native setup, tuning, or execution failed; no fallback is substituted."""


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise BackendError("The application suite requires PyTorch; use a PyTorch environment.") from exc


def _dtype(torch, value):
    aliases = {
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }
    if isinstance(value, str):
        value = aliases.get(value.lower().removeprefix("torch."))
    if value not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be fp32, fp16, or bf16")
    return value


def _library_path(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise BackendError(f"Native suite library does not exist: {path}")
        return path
    env = os.environ.get("MEGAFMM_SUITE_LIBRARY")
    if env:
        return _library_path(env)
    root = Path(__file__).resolve().parents[2]
    # Prefer a dedicated build.  Never select a random stale shared library.
    for folder in ("build-suite", "build"):
        path = root / folder / "libmegafmm_suite.so"
        if path.is_file():
            return path
    raise BackendError(
        "libmegafmm_suite.so was not found. Build the megafmm_suite CMake target "
        "and pass options.library or MEGAFMM_SUITE_LIBRARY."
    )


def fp64_gemm_diagnostics(records, dense_outputs=None, dtype="fp32"):
    """Evaluate stored-input local errors, separately from application quality.

    ``dense_outputs`` must be the independently evaluated dense FP32 outputs
    for these exact stored inputs.  Without them, local historical gates remain
    unchecked: the dtype floor alone is not the historical gate.
    This function performs CPU FP64 work and belongs outside measurement.
    """
    torch = _torch()
    floor = {"fp32": 1e-6, "fp16": 1e-4, "bf16": 1e-3}[dtype]
    records = list(records)
    if dense_outputs is not None and len(dense_outputs) != len(records):
        raise ValueError("dense_outputs must match the recorded GEMM count")
    items = []
    for index, record in enumerate(records):
        a, b = record["a"].double().cpu(), record["b"].double().cpu()
        out = record["out"].double().cpu()
        reference = a @ b
        denominator_squared = max(float(reference.square().sum()), sys.float_info.min)

        def error(value):
            if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(reference).all()):
                return float("inf")
            return math.sqrt(float((value - reference).square().sum()) / denominator_squared)

        actual_error = error(out)
        item = {"index": index, "m": a.shape[0], "n": b.shape[1], "k": a.shape[1],
                "input_dtype": dtype, "reference": "cpu_fp64_from_actual_stored_inputs",
                "relative_error": actual_error if math.isfinite(actual_error) else "+inf",
                "finite": math.isfinite(actual_error), "gemm_gate_pass": None,
                "gate_status": "unchecked_missing_dense_same_input_reference"}
        if dense_outputs is not None:
            dense_error = error(dense_outputs[index].double().cpu())
            limit = 3 * dense_error + floor
            item.update(dense_error=dense_error if math.isfinite(dense_error) else "+inf",
                        gate_limit=limit if math.isfinite(limit) else "+inf",
                        gemm_gate_pass=math.isfinite(actual_error) and math.isfinite(dense_error) and actual_error <= limit,
                        gate_status="checked", gate_formula="eF <= 3*eF_dense + dtype_floor")
        items.append(item)
    checked = bool(items) and dense_outputs is not None
    return {"gemm_gate_pass": all(item["gemm_gate_pass"] is True for item in items) if checked else None,
            "gate_status": "checked" if checked else "unchecked",
            "finite_pass": all(item["finite"] for item in items) if items else None,
            "count": len(items), "dtype_floor": floor, "records": items}


class _NativeLibrary:
    def __init__(self, path):
        self.path = str(path)
        try:
            self.lib = ctypes.CDLL(self.path)
        except OSError as exc:
            raise BackendError(f"Cannot load {self.path}: {exc}") from exc
        specs = {
            "abi_version": ([], ctypes.c_int),
            "last_error": ([], ctypes.c_char_p),
            "runtime_metadata": ([], ctypes.c_char_p),
            "create": ([ctypes.c_int64] * 3 + [ctypes.c_int] * 2 +
                       [ctypes.c_uint64] + [ctypes.c_int] * 5, ctypes.c_void_p),
            "workspace": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "tune": ([ctypes.c_void_p] * 6, ctypes.c_int),
            "run": ([ctypes.c_void_p] * 6, ctypes.c_int),
            "metadata": ([ctypes.c_void_p], ctypes.c_char_p),
            "destroy": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (args, result) in specs.items():
            fn = getattr(self.lib, "megafmm_suite_" + name)
            fn.argtypes, fn.restype = args, result
            setattr(self, name, fn)
        if self.abi_version() != 1:
            raise BackendError("Unsupported native suite ABI; rebuild megafmm_suite")

    def error(self, action):
        message = self.last_error()
        return BackendError(f"{action}: {message.decode() if message else 'native error'}")

    def check(self, result, action):
        if result != 0:
            raise self.error(action)

    def describe(self, handle):
        raw = self.metadata(handle)
        if raw is None:
            raise self.error("Read native plan metadata")
        return json.loads(raw)


class Backend:
    """GEMM backend with shape-specific plans prepared outside measurement.

    Run an application once, then call ``freeze()`` before ordinary timing or
    graph capture.  An unseen shape after freeze is an error.  ``setup`` also
    accepts a pair or iterable of input pairs for explicit preparation.

    All inputs must already be on ``device``.  Dtype conversion and contiguous
    materialization occur inside ``gemm``, so their costs are measured.
    The optional recording mode copies actual rounded inputs and outputs to
    CPU for FP64 local diagnostics; use it outside timing and graph capture.
    A Backend owns mutable workspace and must not be shared concurrently across
    streams or threads.  Independent Backend instances provide independent plans.
    """
    NATIVE = {"cublaslt": 0, "megafmm": 1, "megafmm_fixed": 2}

    def __init__(self, name, device="cuda", dtype="fp32", options=None):
        self.torch = torch = _torch()
        self.name = str(name)
        if self.name not in (*self.NATIVE, "torch", "practical_torch"):
            raise ValueError(f"Unknown GEMM backend: {name}")
        self.options = dict(options or {})
        if not 0 <= int(self.options.get("workspace_limit", 8 << 30)) < (1 << 64):
            raise ValueError("workspace_limit must be a nonnegative uint64 byte count")
        if not 1 <= int(self.options.get("heuristic_candidates", 32)) <= 128:
            raise ValueError("heuristic_candidates must be between 1 and 128")
        if not 1 <= int(self.options.get("tuning_input_sets", 1)) <= 4:
            raise ValueError("tuning_input_sets must be between 1 and 4")
        if int(self.options.get("sw_tuning_candidates", 0)) < 0:
            raise ValueError("sw_tuning_candidates must be nonnegative")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = _dtype(torch, dtype)
        self.dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[self.dtype]
        self._plans = {}
        self._failed_plans = []
        self._torch_shapes = set()
        self._frozen = False
        self._closed = False
        self.record_inputs = bool(self.options.get("record_inputs", False))
        self.record_limit = int(self.options.get("record_limit", 64))
        if self.record_limit < 1:
            raise ValueError("record_limit must be positive")
        self.recorded = []
        self.recorded_dropped = 0
        self._native = None
        self._runtime_metadata = {}
        if self.name in self.NATIVE:
            if self.device.type != "cuda":
                raise ValueError("Native MegaFMM/cuBLASLt backends require a CUDA device")
            with torch.cuda.device(self.device):
                self._native = _NativeLibrary(_library_path(self.options.get("library")))
                raw = self._native.runtime_metadata()
                if raw is None:
                    raise self._native.error("Read loaded CUDA/cuBLAS versions")
                self._runtime_metadata = json.loads(raw)
        self._allow_tf32 = bool(self.options.get("allow_tf32", self.name == "practical_torch"))
        if self.name == "torch" and self._allow_tf32:
            raise ValueError("The matched torch backend requires allow_tf32=False; use practical_torch")

    def _capturing(self):
        return self.device.type == "cuda" and self.torch.cuda.is_current_stream_capturing()

    def _prepare_inputs(self, a, b):
        if self._closed:
            raise BackendError("Backend is closed")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError(f"GEMM requires compatible 2D inputs, got {tuple(a.shape)} and {tuple(b.shape)}")
        if a.device != self.device or b.device != self.device:
            raise ValueError(f"Stage inputs on {self.device} before calling gemm")
        return a.to(dtype=self.dtype).contiguous(), b.to(dtype=self.dtype).contiguous()

    def _key(self, a, b):
        return (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]), self.dtype_name, str(self.device))

    def _create_plan(self, key, a, b, out):
        native, torch = self._native, self.torch
        start = time.perf_counter()
        m, n, k = key[:3]
        handle = native.create(
            m, n, k, {"bf16": 0, "fp16": 1, "fp32": 2}[self.dtype_name],
            self.NATIVE[self.name], int(self.options.get("workspace_limit", 8 << 30)),
            int(self.options.get("heuristic_candidates", 32)),
            int(self.options.get("tuning_input_sets", 1)),
            int(self.options.get("sw_tuning_candidates", 0)),
            int(bool(self.options.get("sw_tuning_graph", False))),
            int(not bool(self.options.get("strict", False))),
        )
        if handle is None:
            raise native.error(f"Create {self.name} plan {key}")
        workspace = None
        try:
            size = ctypes.c_uint64()
            native.check(native.workspace(handle, ctypes.byref(size)), "Query plan workspace")
            workspace_start = time.perf_counter()
            workspace = torch.empty(size.value, dtype=torch.uint8, device=self.device)
            allocation_ms = (time.perf_counter() - workspace_start) * 1000
            stream = torch.cuda.current_stream(self.device).cuda_stream
            ptr = workspace.data_ptr() if size.value else 0
            native.check(native.tune(handle, a.data_ptr(), b.data_ptr(), out.data_ptr(), ptr, stream),
                         f"Tune {self.name} plan {key}")
            description = native.describe(handle)
            description.update(workspace_allocation_ms=allocation_ms,
                               setup_ms=(time.perf_counter() - start) * 1000)
            self._plans[key] = {"handle": handle, "workspace": workspace,
                                "metadata": description}
            return self._plans[key]
        except Exception as exc:
            # Strict tuning failure still contains candidate error/timing records.
            try:
                description = native.describe(handle)
            except Exception as metadata_exc:
                description = {"metadata_error": str(metadata_exc)}
            description.update(error=str(exc), shape=list(key[:3]), dtype=self.dtype_name,
                               setup_ms=(time.perf_counter() - start) * 1000)
            self._failed_plans.append(description)
            native.destroy(handle)
            raise

    def gemm(self, a, b):
        torch = self.torch
        device_context = torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), device_context:
            a, b = self._prepare_inputs(a, b)
            key = self._key(a, b)
            capturing = self._capturing()
            if self.record_inputs and capturing:
                raise BackendError("Input recording must be disabled during CUDA Graph capture")
            if self._native is None:
                if self._frozen and key not in self._torch_shapes:
                    raise BackendError(f"Unprepared GEMM shape after freeze: {key}")
                self._torch_shapes.add(key)
                # TF32 is a process setting.  Restore it after this call so
                # independent matched/practical baselines do not contaminate one another.
                if self.device.type == "cuda":
                    old_tf32 = torch.backends.cuda.matmul.allow_tf32
                    old_reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                    old_bf16_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = self._allow_tf32
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                        out = torch.mm(a, b).float()
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_reduced
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_bf16_reduced
                else:
                    out = torch.mm(a, b).float()
            else:
                out = torch.empty((key[0], key[1]), dtype=torch.float32, device=self.device)
                plan = self._plans.get(key)
                if plan is None:
                    if self._frozen or capturing:
                        raise BackendError(f"Unprepared GEMM shape; run setup before freeze/capture: {key}")
                    plan = self._create_plan(key, a, b, out)
                workspace = plan["workspace"]
                stream = torch.cuda.current_stream(self.device)
                # Native launches are invisible to PyTorch's dispatcher.  Tell
                # its allocator about stream use before temporary casts/views
                # or a caller-owned input can be released.
                a.record_stream(stream)
                b.record_stream(stream)
                if workspace.numel():
                    workspace.record_stream(stream)
                self._native.check(self._native.run(
                    plan["handle"], a.data_ptr(), b.data_ptr(), out.data_ptr(),
                    workspace.data_ptr() if workspace.numel() else 0,
                    stream.cuda_stream), f"Execute {self.name} GEMM")
            if self.record_inputs:
                if len(self.recorded) < self.record_limit:
                    self.recorded.append({"a": a.detach().cpu().clone(), "b": b.detach().cpu().clone(),
                                          "out": out.detach().cpu().clone()})
                else:
                    self.recorded_dropped += 1
            return out

    def setup(self, inputs=None):
        """Prepare plans from actual input pairs; application warmup is equivalent."""
        if inputs is None:
            return self
        if (isinstance(inputs, (tuple, list)) and len(inputs) == 2 and
                all(hasattr(x, "shape") for x in inputs)):
            inputs = [inputs]
        for a, b in inputs:
            self.gemm(a, b)
        return self

    def freeze(self):
        """Forbid new plans/shapes during steady-state timing and capture."""
        self._frozen = True
        return self

    def clear_recording(self):
        self.recorded.clear()
        self.recorded_dropped = 0

    def local_gemm_diagnostics(self, dense_backend=None):
        """Diagnose recorded GEMMs; optional same-input cuBLASLt historical gate.

        Missing recordings or a missing dense reference are explicitly
        unchecked.  A recording limit is reported as sampled coverage.
        ``dense_backend`` may create/tune plans here, outside application timing.
        """
        if self._capturing():
            raise BackendError("FP64 diagnostics must run outside CUDA Graph capture")
        dense_outputs = None
        if dense_backend is not None:
            if dense_backend.name != "cublaslt" or dense_backend.dtype != self.dtype:
                raise ValueError("Historical local gates require a same-dtype native cublaslt backend")
            if dense_backend.record_inputs:
                raise ValueError("Disable recording on the diagnostic dense backend")
            dense_outputs = []
            for record in self.recorded:
                a = record["a"].to(dense_backend.device)
                b = record["b"].to(dense_backend.device)
                dense_outputs.append(dense_backend.gemm(a, b).detach().cpu())
        result = fp64_gemm_diagnostics(self.recorded, dense_outputs, self.dtype_name)
        result.update(coverage="sampled" if self.recorded_dropped else "all_recorded_calls",
                      dropped=self.recorded_dropped,
                      dense_reference="same_input_independently_tuned_cublaslt" if dense_backend else None)
        return result

    def metadata(self):
        torch = self.torch
        native = self._native is not None
        plans = []
        for plan in self._plans.values():
            description = dict(plan["metadata"])
            # Metadata is only called outside timing/capture; latest binding may
            # reveal an actual alignment-dependent execution route.
            if not self._closed:
                description.update(self._native.describe(plan["handle"]))
            plans.append(description)
        return {
            "name": self.name, "device": str(self.device), "input_dtype": self.dtype_name,
            "accumulation_dtype": "fp32", "output_dtype": "fp32",
            "output_rounding": "fp32" if native or self.dtype == torch.float32 else f"{self.dtype_name}_then_fp32",
            "precision_policy": "strict_historical_gate" if native and self.options.get("strict") else
                                "finite_candidates_application_gate" if self.name == "megafmm" else "application_gate",
            "allow_tf32": self._allow_tf32 if not native else False,
            "torch_reduced_precision_reduction": False if not native else None,
            "layout": "row_major_contiguous_materialized_inside_gemm",
            "data_staging": "outside_gemm", "frozen": self._frozen, "closed": self._closed,
            "library": self._native.path if native else None,
            "implementation": "public_megafmm_plan_api" if native else "torch.mm_framework_dispatch",
            "torch_version": str(torch.__version__), "torch_build_cuda": torch.version.cuda,
            "loaded_native_versions": dict(self._runtime_metadata),
            "plans": plans, "failed_plans": list(self._failed_plans),
            "framework_shapes": [list(shape) for shape in sorted(self._torch_shapes)],
            "plan_count": len(plans),
            "workspace_bytes": sum(p["workspace_bytes"] for p in plans),
            "plan_create_ms": sum(p["plan_create_ms"] for p in plans),
            "tune_ms": sum(p["tune_ms"] for p in plans),
            "setup_ms": sum(p["setup_ms"] for p in plans),
            "recorded_count": len(self.recorded), "recorded_dropped": self.recorded_dropped,
        }

    def close(self):
        if self._closed:
            return
        errors = []
        if self._native is not None:
            with self.torch.cuda.device(self.device):
                for plan in self._plans.values():
                    try:
                        # Save diagnostics before releasing handles/workspace.
                        plan["metadata"].update(self._native.describe(plan["handle"]))
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        result = self._native.destroy(plan["handle"])
                        if result:
                            errors.append(self._native.error("Destroy native plan"))
                        plan["handle"] = None
                        plan["workspace"] = None
        self._closed = True
        if errors:
            raise BackendError("; ".join(str(error) for error in errors))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
