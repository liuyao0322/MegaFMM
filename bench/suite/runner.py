"""Process-isolated experimental runner. Importing this module never imports torch.

Schema version 1 deliberately accepts explicit cases rather than implicit grids.
Raw group observations are preserved; independent processes are the statistical unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
APPLICATIONS = {"matrix_function", "muon", "mlp", "purification", "linpack", "covariance"}
BACKENDS = {"cublaslt", "megafmm", "megafmm_fixed", "torch", "practical_torch"}
TOP_FIELDS = {"schema_version", "suite", "description", "backends", "baseline", "modes",
              "measurement", "backend_options", "cases"}
MEASUREMENT_DEFAULTS = {"independent_processes": 3, "warmup": 3, "groups": 5,
                        "calls_per_group": 10, "fresh_quality_trials": 2,
                        "rotate_inputs": 2, "timeout_seconds": 900, "seed": 20261021}
CASE_FIELDS = {"id", "kind", "app", "application", "description", "dtype", "n", "m", "k",
               "shape", "iterations", "function", "condition_number", "regularization",
               "occupied", "gap", "tokens", "hidden", "intermediate", "activation",
               "input_path", "input_scope", "asset_description", "quality", "evaluation",
               "algorithm", "dynamic_range", "distribution", "backends", "modes", "max_iterations",
               "stop_tolerance", "check_every", "nocc", "bias", "gelu_approximate", "quality_thresholds",
               "trace_stop_tolerance", "input_scale", "block_size", "transition_radius"}
OPTION_FIELDS = {"workspace_limit", "heuristic_candidates", "tuning_input_sets",
                 "sw_tuning_candidates", "sw_tuning_graph", "strict", "record_inputs", "record_limit", "allow_tf32"}


def case_kind(case):
    return case.get("app", case.get("application", case.get("kind")))


def resolved_backend_options(payload):
    """Tuning follows the measured mode; input recording never enters timed calls."""
    options = dict(payload.get("backend_options", {}))
    options["sw_tuning_graph"] = payload["mode"] == "graph"
    options["record_inputs"] = False
    if payload.get("library"):
        options["library"] = payload["library"]
    return options


class ConfigurationError(ValueError):
    pass


def _check_keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be an object")
    extra = set(value) - allowed
    if extra:
        raise ConfigurationError(f"unknown {context} field(s): {', '.join(sorted(extra))}")


def _positive(value, context, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ConfigurationError(f"{context} must be {'nonnegative' if allow_zero else 'positive'} integer")


def _name_list(value, allowed, context):
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value) or \
            len(value) != len(set(value)) or set(value) - allowed:
        raise ConfigurationError(f"{context} must be a nonempty unique list from {sorted(allowed)}")


def validate_config(raw):
    _check_keys(raw, TOP_FIELDS, "config")
    if type(raw.get("schema_version")) is not int or raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(raw.get("suite"), str) or not raw["suite"]:
        raise ConfigurationError("suite must be a nonempty string")
    cfg = dict(raw)
    cfg.setdefault("backends", ["cublaslt", "megafmm"])
    cfg.setdefault("baseline", cfg["backends"][0] if cfg["backends"] else None)
    cfg.setdefault("modes", ["launch", "graph"])
    for names, allowed, context in [(cfg["backends"], BACKENDS, "backends"),
                                      (cfg["modes"], {"launch", "graph"}, "modes")]:
        _name_list(names, allowed, context)
    if cfg["baseline"] not in cfg["backends"]:
        raise ConfigurationError("baseline must occur in backends")
    _check_keys(cfg.get("measurement", {}), set(MEASUREMENT_DEFAULTS), "measurement")
    cfg["measurement"] = {**MEASUREMENT_DEFAULTS, **cfg.get("measurement", {})}
    for key, value in cfg["measurement"].items():
        _positive(value, f"measurement.{key}", key in {"warmup", "seed"})
    _check_keys(cfg.get("backend_options", {}), OPTION_FIELDS, "backend_options")
    cfg.setdefault("backend_options", {})
    if "workspace_limit" in cfg["backend_options"]:
        _positive(cfg["backend_options"]["workspace_limit"], "workspace_limit", True)
    for key, value in cfg["backend_options"].items():
        if key in {"sw_tuning_graph", "strict", "record_inputs", "allow_tf32"}:
            if not isinstance(value, bool):
                raise ConfigurationError(f"backend_options.{key} must be boolean")
        elif key != "workspace_limit":
            _positive(value, f"backend_options.{key}", key == "sw_tuning_candidates")
    for key, bound in (("heuristic_candidates", 128), ("tuning_input_sets", 4)):
        if cfg["backend_options"].get(key, 1) > bound:
            raise ConfigurationError(f"backend_options.{key} must be <= {bound}")
    if cfg["backend_options"].get("workspace_limit", 0) >= (1 << 64):
        raise ConfigurationError("workspace_limit must fit uint64")
    if not isinstance(cfg.get("cases"), list) or not cfg["cases"]:
        raise ConfigurationError("cases must be a nonempty explicit list")
    seen = set()
    for case in cfg["cases"]:
        _check_keys(case, CASE_FIELDS, "case")
        ident = case.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", ident) or ident in seen:
            raise ConfigurationError("case id must be unique and contain only letters, digits, _, ., -")
        seen.add(ident)
        kind = case_kind(case)
        if not isinstance(kind, str) or kind not in APPLICATIONS | {"gemm"}:
            raise ConfigurationError(f"case {ident}: unknown kind {kind!r}")
        if not isinstance(case.get("dtype", "fp32"), str) or case.get("dtype", "fp32") not in {"fp32", "fp16", "bf16"}:
            raise ConfigurationError(f"case {ident}: dtype must be fp32/fp16/bf16")
        if case.get("evaluation", "fixed_work") not in {"fixed_work", "time_to_quality"}:
            raise ConfigurationError(f"case {ident}: unknown evaluation")
        if case.get("evaluation") == "time_to_quality" and kind not in {"matrix_function", "purification"}:
            raise ConfigurationError(f"case {ident}: time_to_quality supports matrix_function/purification only")
        for field in ("m", "n", "k", "iterations", "tokens", "hidden", "intermediate", "occupied", "nocc", "max_iterations", "check_every", "block_size"):
            if field in case:
                _positive(case[field], f"case {ident}.{field}", field in {"occupied", "nocc"})
        if "shape" in case and (not isinstance(case["shape"], list) or not case["shape"] or
                                  any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in case["shape"])):
            raise ConfigurationError(f"case {ident}.shape must contain positive integer dimensions")
        if not isinstance(case.get("quality", {}), dict):
            raise ConfigurationError(f"case {ident}.quality must be an object")
        for name in ("quality", "quality_thresholds"):
            if name in case:
                if not isinstance(case[name], dict):
                    raise ConfigurationError(f"case {ident}.{name} must be an object")
                for metric, value in case[name].items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ConfigurationError(f"case {ident}.{name}.{metric} must be finite and nonnegative")
        for field in ("condition_number", "regularization", "gap", "dynamic_range", "input_scale", "stop_tolerance", "trace_stop_tolerance"):
            if field in case and (isinstance(case[field], bool) or not isinstance(case[field], (int, float)) or not math.isfinite(case[field]) or case[field] < 0):
                raise ConfigurationError(f"case {ident}.{field} must be finite and nonnegative")
        if "shape" in case:
            raise ConfigurationError(f"case {ident}: use explicit m,n,k or tokens,hidden,intermediate; shape shorthand is not supported")
        if "input_path" in case and not isinstance(case["input_path"], str):
            raise ConfigurationError(f"case {ident}.input_path must be a string")
        if "bias" in case and not isinstance(case["bias"], bool):
            raise ConfigurationError(f"case {ident}.bias must be boolean")
        if "gelu_approximate" in case and case["gelu_approximate"] not in {"none", "tanh"}:
            raise ConfigurationError(f"case {ident}.gelu_approximate must be none/tanh")
        if case.get("distribution", "normal") not in {"normal", "uniform"}:
            raise ConfigurationError(f"case {ident}.distribution must be normal/uniform")
        for name in ("condition_number", "dynamic_range"):
            if name in case and case[name] < 1:
                raise ConfigurationError(f"case {ident}.{name} must be >= 1")
        for name in ("stop_tolerance", "trace_stop_tolerance"):
            if name in case and case[name] <= 0:
                raise ConfigurationError(f"case {ident}.{name} must be > 0")
        if kind not in {"muon", "mlp"} and any(k in case for k in ("dynamic_range", "input_scale", "distribution")):
            raise ConfigurationError(f"case {ident}: dynamic_range/input_scale/distribution apply to muon/mlp only")
        if case.get("input_scope") == "external" and not case.get("input_path"):
            raise ConfigurationError(f"case {ident}: external input requires input_path")
        for key, allowed in (("backends", BACKENDS), ("modes", {"launch", "graph"})):
            if key in case:
                _name_list(case[key], allowed, f"case {ident}.{key}")
        if cfg["baseline"] not in case.get("backends", cfg["backends"]):
            raise ConfigurationError(f"case {ident}: baseline must occur in selected backends")
        if "block_size" in case and kind != "linpack":
            raise ConfigurationError(f"case {ident}: block_size applies to linpack only")
        if "transition_radius" in case:
            radius = case["transition_radius"]
            if kind != "covariance" or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius < 1:
                raise ConfigurationError(f"case {ident}: covariance transition_radius must lie in (0,1)")
        if kind == "covariance" and case.get("dtype", "fp32") != "fp32":
            raise ConfigurationError(f"case {ident}: covariance requires fp32")
        if kind == "linpack":
            if case.get("dtype", "fp32") != "fp32":
                raise ConfigurationError(f"case {ident}: linpack requires fp32; this is not FP64 HPL")
            if case.get("modes", cfg["modes"]) != ["launch"]:
                raise ConfigurationError(f"case {ident}: linpack requires launch mode for pivot/status synchronization")
            if any(key in case for key in ("iterations", "max_iterations", "check_every")):
                raise ConfigurationError(f"case {ident}: linpack performs one solve per call; use n/block_size")
        if kind == "gemm":
            if not all(field in case for field in ("m", "n", "k")):
                raise ConfigurationError(f"case {ident}: GEMM requires m,n,k")
            if case.get("algorithm", "sw-tuned") not in {"dense", "auto", "sw", "tuned", "sw-tuned", "sw-tuned-strict", "sw-tuned-diagnostic"}:
                raise ConfigurationError(f"case {ident}: unknown GEMM algorithm")
            if set(case.get("backends", cfg["backends"])) != {"cublaslt", "megafmm"}:
                raise ConfigurationError(f"case {ident}: native GEMM binary compares cublaslt/megafmm only")
    return cfg


def load_config(path):
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
    cfg = validate_config(raw)
    for case in cfg["cases"]:
        if case.get("input_path"):
            path = Path(os.path.expandvars(case["input_path"])).expanduser()
            case["input_path"] = str((path if path.is_absolute() else ROOT / path).resolve())
    return cfg


def _json(value):
    # Refuse non-standard NaN/Infinity in permanent records; preserve their meaning explicitly.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(data), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _command(args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_hashes():
    files = []
    for folder in ("solution", "include", "bench/suite"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in {".py", ".cpp", ".cu", ".cuh", ".h", ".inc", ".json"})
    files.extend(ROOT / p for p in ("CMakeLists.txt", "bench/benchmark.cu", "scripts/run_suite.py", "scripts/report_suite.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}


def input_hashes(config):
    return {c["input_path"]: sha256(c["input_path"]) for c in config["cases"] if c.get("input_path")}


def resolved_library(explicit):
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    if os.environ.get("MEGAFMM_SUITE_LIBRARY"):
        return str(Path(os.environ["MEGAFMM_SUITE_LIBRARY"]).expanduser().resolve())
    for folder in ("build-suite", "build"):
        path = ROOT / folder / "libmegafmm_suite.so"
        if path.is_file():
            return str(path.resolve())
    return None


def manifest(config, args):
    build_manifest_path = Path(args.library).parent / "build_manifest.json" if args.library else None
    build_manifest = json.loads(build_manifest_path.read_text()) if build_manifest_path and build_manifest_path.is_file() else None
    return {"schema_version": SCHEMA_VERSION, "suite": config["suite"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "git_status": _command(["git", "status", "--porcelain"]),
            "source_hashes": source_hashes(), "input_asset_hashes": input_hashes(config),
            "library": str(args.library) if args.library else None,
            "library_sha256": sha256(args.library) if args.library else None,
            "native_build_manifest": build_manifest,
            "gemm_binary": str(args.gemm_binary) if args.gemm_binary else None,
            "gemm_binary_sha256": sha256(args.gemm_binary) if args.gemm_binary else None,
            "python": sys.version, "platform": platform.platform(), "device": args.device,
            "selected_case_ids": [c["id"] for c in config["cases"] if not args.case or c["id"] in args.case],
            "environment": {key: os.environ.get(key) for key in
                            ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "SLURM_JOB_ID", "SLURM_JOB_GPUS",
                             "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
            "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"])
            if args.device.startswith("cuda") else None,
            "statistical_unit": "independent_process", "sample_unit": "group_mean",
            "quality_policy": "separate implementation, local GEMM, application quality, and speed gates",
            "scope": "fixed-work and synchronized time-to-quality algorithms; synthetic inputs do not establish full task quality"}


def runtime_fingerprint(device):
    """Identify the actual device and dependency build, including across resume."""
    script = '''import json,sys
import torch
d=torch.device(sys.argv[1])
info={"torch":str(torch.__version__),"cuda_runtime":torch.version.cuda,"python":sys.version}
if d.type=="cuda":
 p=torch.cuda.get_device_properties(d)
 info.update(gpu=p.name,uuid=str(getattr(p,"uuid","unavailable")),capability=[p.major,p.minor],total_memory=p.total_memory)
print(json.dumps(info,sort_keys=True))'''
    raw = _command([sys.executable, "-s", "-c", script, device])
    if raw is None:
        raise ConfigurationError("cannot identify PyTorch/device runtime; check the selected Python environment")
    return {"runtime": json.loads(raw), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) if device.startswith("cuda") else None}


def _base_record(payload, backend, **kwargs):
    return {"schema_version": SCHEMA_VERSION, "case_id": payload["case"]["id"],
            "kind": case_kind(payload["case"]),
            "backend": backend, "mode": payload["mode"], "process_index": payload["process_index"],
            "seed": payload["seed"], "worker_pid": os.getpid(),
            "dtype": payload["case"].get("dtype", "fp32"), "device": payload["device"],
            "requested_backend_options": dict(payload.get("backend_options", {})),
            "resolved_backend_options": resolved_backend_options(payload),
            "tuning_mode_policy": "measurement mode overrides explicit sw_tuning_graph",
            "implementation_pass": False, "gemm_gate_pass": None,
            "application_quality_pass": None, "performance_win": None, "qualified_win": None,
            **kwargs}


def _mark_failure(payload, name, exc, status=None):
    message = str(exc)
    if status is None:
        if isinstance(exc, FileNotFoundError):
            status = "skipped" if payload["case"].get("input_path") else "failed"
        elif isinstance(exc, NotImplementedError) or "not supported" in message.lower():
            status = "unsupported"
        elif "out of memory" in message.lower():
            status = "oom"
        else:
            status = "failed"
    return _base_record(payload, name, status=status, error=message,
                        error_type=type(exc).__name__, traceback=traceback.format_exc())


def _time_summary(samples):
    result = {}
    for key in ("event_ms", "wall_ms"):
        values = [s[key] for s in samples if type(s.get(key)) in (float, int) and math.isfinite(s[key]) and s[key] > 0]
        if not values:
            result[f"steady_{key}"] = None
            continue
        result[f"steady_{key}"] = statistics.median(values)
        result[f"mean_{key}"] = statistics.mean(values)
        result[f"cv_{key}"] = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) else 0.
    result["quantile_unit"] = "group_mean"
    result["group_count"] = len(samples)
    return result


def _finalize_pairs(result, baseline):
    by_name = {run["backend"]: run for run in result["runs"]}
    base = by_name.get(baseline, {})
    # Python dispatch, required layout conversions and host stopping decisions
    # are part of an application invocation. Native GEMM retains event timing.
    time_key = "steady_wall_ms" if base.get("kind") != "gemm" and base.get("steady_wall_ms") is not None else "steady_event_ms"
    for run in result["runs"]:
        if run["backend"] == baseline or run.get("status") != "ok" or base.get("status") != "ok":
            continue
        bt, rt = base.get(time_key), run.get(time_key)
        if type(bt) not in (int, float) or type(rt) not in (int, float) or not math.isfinite(bt) or not math.isfinite(rt) or bt <= 0 or rt <= 0:
            continue
        run["speedup"] = bt / rt
        run["speedup_time_basis"] = time_key
        run["performance_win"] = bt > rt
        # Native GEMM has no independent application metric: use its explicit local numerical gate.
        quality = run.get("application_quality_pass")
        if run.get("kind") == "gemm":
            quality = run.get("gemm_gate_pass")
        baseline_quality = base.get("application_quality_pass") if run.get("kind") != "gemm" else base.get("implementation_pass")
        run["qualified_win"] = bool(run["performance_win"] and quality is True and baseline_quality is True and
                                     run.get("implementation_pass") is True and base.get("implementation_pass") is True)
        run["baseline_quality_pass"] = baseline_quality
        run["baseline_implementation_pass"] = base.get("implementation_pass")
        run["baseline_status"] = base.get("status")
        run["baseline_backend"] = baseline
        delta = bt - rt
        bc, rc = base.get("cold_setup_ms"), run.get("cold_setup_ms")
        run["break_even_calls"] = math.ceil(max(0, rc - bc) / delta) if delta > 0 and isinstance(bc, (int, float)) and isinstance(rc, (int, float)) else None


def application_worker(payload):
    import torch
    from bench.suite.applications import create_application
    from bench.suite.backends import Backend

    result = {"runs": [], "samples": [], "metrics": [], "environment": {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda}}
    case, measure = payload["case"], payload["measurement"]
    dynamic = case.get("evaluation") == "time_to_quality"
    if (dynamic or case_kind(case) == "linpack") and payload["mode"] == "graph":
        return {"runs": [_mark_failure(payload, name, NotImplementedError("application requires synchronized host decisions; CUDA Graph is unsupported"), "unsupported")
                         for name in payload["backends"]], "samples": [], "metrics": []}
    def execute(app, backend):
        return app.execute_to_quality(backend) if dynamic else app.execute(backend)
    device = torch.device(payload["device"])
    cuda = device.type == "cuda"
    if cuda:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        result["environment"]["gpu"] = torch.cuda.get_device_name(device)
        result["environment"]["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        result["environment"]["gpu_uuid"] = str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable"))
        result["environment"]["cuda_device_index"] = torch.cuda.current_device()
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    names = payload["backends"]
    live = {}
    rng = random.Random(payload["seed"])
    setup_order = list(names)
    rng.shuffle(setup_order)
    with torch.no_grad():
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        prepare_start = time.perf_counter()
        # Share immutable scientific inputs and independent FP64 references.
        # Each backend still owns its mutable plans, workspaces and outputs.
        shared_apps = [create_application(case, str(device), payload["seed"] + 1000 + i)
                       for i in range(measure["rotate_inputs"])]
        shared_quality_apps = [create_application(case, str(device), payload["seed"] + 100000 + i)
                               for i in range(measure["fresh_quality_trials"])]
        shared_prepare_ms = (time.perf_counter() - prepare_start) * 1000
        for name in setup_order:
            backend = None
            try:
                start = time.perf_counter()
                # Every slot persists through capture and measurement; no captured input is mutated.
                apps = shared_apps
                input_prepare_ms = shared_prepare_ms
                options = resolved_backend_options(payload)
                start = time.perf_counter()
                backend = Backend(name, str(device), case.get("dtype", "fp32"), options)
                sync()
                backend_setup_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                execute(apps[0], backend)  # Discover all plans, tune outside capture.
                sync()
                plan_tune_warmup_ms = (time.perf_counter() - start) * 1000
                backend.freeze()
                metrics = []
                diagnostics = []
                for trial, quality_app in enumerate(shared_quality_apps):
                    ref = quality_app.make_reference()
                    record_gemm = bool(payload["backend_options"].get("record_inputs")) and trial == 0
                    backend.clear_recording()
                    backend.record_inputs = record_gemm
                    output = execute(quality_app, backend)
                    backend.record_inputs = False
                    sync()
                    values = quality_app.validate(output, ref)
                    if "quality_pass" not in values:
                        raise RuntimeError("application validate() omitted quality_pass")
                    metrics.append(values)
                    result["metrics"].append({"case_id": case["id"], "backend": name,
                        "mode": payload["mode"], "process_index": payload["process_index"],
                        "trial": trial, "input": quality_app.metadata(), "values": values})
                    if record_gemm:
                        dense_diagnostic = None
                        try:
                            if cuda and name in {"megafmm", "megafmm_fixed", "cublaslt"}:
                                dense_diagnostic = Backend("cublaslt", str(device), case.get("dtype", "fp32"), options)
                            diag = backend.local_gemm_diagnostics(dense_diagnostic)
                            diagnostics.append(diag)
                            result["metrics"].append({"case_id": case["id"], "backend": name,
                                "mode": payload["mode"], "process_index": payload["process_index"],
                                "phase": "local_gemm_diagnostics", "values": diag})
                        finally:
                            if dense_diagnostic is not None:
                                dense_diagnostic.close()
                calls = []
                captures = []
                outputs = [None] * len(apps)
                captured_ms = 0.
                start = time.perf_counter()
                if payload["mode"] == "graph":
                    if not cuda:
                        raise NotImplementedError("CUDA Graph is unavailable on CPU")
                    for app in apps:
                        app.execute(backend)
                        sync()
                        graph = torch.cuda.CUDAGraph()
                        capture_start = time.perf_counter()
                        with torch.cuda.graph(graph):
                            graph_output = app.execute(backend)
                        sync()
                        captured_ms += (time.perf_counter() - capture_start) * 1000
                        captures.append((graph, graph_output))
                        calls.append(graph.replay)
                        outputs[len(captures)-1] = graph_output
                else:
                    for index, app in enumerate(apps):
                        def invoke(app=app, backend=backend, index=index, outputs=outputs):
                            outputs[index] = execute(app, backend)
                        calls.append(invoke)
                sync()
                graph_prepare_ms = (time.perf_counter() - start) * 1000 if payload["mode"] == "graph" else 0.
                for i in range(measure["warmup"]):
                    calls[i % len(calls)]()
                sync()
                backend_metadata = backend.metadata()
                record = _base_record(payload, name, status="ok", implementation_pass=True,
                    application_quality_pass=all(x["quality_pass"] is True for x in metrics),
                    input_prepare_ms=input_prepare_ms, backend_setup_ms=backend_setup_ms,
                    plan_tune_warmup_ms=plan_tune_warmup_ms, capture_ms=captured_ms,
                    graph_prepare_ms=graph_prepare_ms,
                    cold_setup_ms=backend_setup_ms + plan_tune_warmup_ms + graph_prepare_ms,
                    tune_ms=backend_metadata.get("tune_ms"), plan_create_ms=backend_metadata.get("plan_create_ms"),
                    setup_ms=backend_metadata.get("setup_ms"),
                    setup_cost_note="cold setup includes constructor, discovery invocation, and capture; native plan/tune costs additionally reported separately",
                    application=apps[0].metadata(), input_sets=[app.metadata() for app in apps],
                    input_hash=hashlib.sha256(json.dumps([app.metadata().get("stored_inputs", {}) for app in apps], sort_keys=True).encode()).hexdigest(),
                    validation_seeds=[payload["seed"] + 100000 + j for j in range(measure["fresh_quality_trials"])],
                    backend_metadata=backend_metadata, measurement=measure,
                    actual_route=[p.get("actual_route") for p in backend_metadata.get("plans", [])] or backend_metadata.get("implementation"),
                    library_sha256=sha256(backend_metadata.get("library")) if backend_metadata.get("library") else None,
                    graph_reset_contract="each replay executes a functional fixed-work application from immutable captured inputs",
                    timing_scope="complete captured CUDA application graph, Python replay launch included in wall time" if payload["mode"] == "graph" else "complete Python-dispatched application including CUDA operations and required casts/checks",
                    evaluation=case.get("evaluation", "fixed_work"),
                    primary_time_basis="steady_wall_ms",
                    memory_contract="shared immutable input/reference slots; independent backend output and plan workspaces; all backends co-resident for interleaved timing",
                    scope=case.get("input_scope", "synthetic_algorithm"))
                if diagnostics:
                    record["gemm_gate_pass"] = all(d["gemm_gate_pass"] is True for d in diagnostics) if all(d["gemm_gate_pass"] is not None for d in diagnostics) else None
                    record["local_gemm_diagnostics"] = diagnostics
                result["runs"].append(record)
                live[name] = {"backend": backend, "calls": calls, "captures": captures, "apps": apps,
                              "outputs": outputs, "record": record}
                # Validate the actual captured/warmup execution path, independently for every persistent input slot.
                for index, app in enumerate(apps):
                    calls[index]()
                    sync()
                    values = app.validate(outputs[index], app.make_reference())
                    record["application_quality_pass"] &= values["quality_pass"] is True
                    result["metrics"].append({"case_id": case["id"], "backend": name, "mode": payload["mode"],
                        "process_index": payload["process_index"], "phase": "timed_path_before", "input_slot": index,
                        "input": app.metadata(), "values": values})
            except Exception as exc:
                failed = _mark_failure(payload, name, exc)
                if backend is not None:
                    backend.record_inputs = False
                    try:
                        failed["backend_metadata"] = backend.metadata()
                        backend.close()
                    except Exception as close_exc:
                        failed["cleanup_error"] = str(close_exc)
                if name in live:
                    live.pop(name)
                    result["runs"] = [r for r in result["runs"] if r["backend"] != name]
                result["runs"].append(failed)
        # Release fresh validation tensors and CPU diagnostic recordings before
        # timing; rotation slots and cached references stay shared and resident.
        quality_app = ref = output = None
        shared_quality_apps.clear()
        for state in live.values():
            state["backend"].clear_recording()
        sync()
        if cuda:
            result["environment"]["setup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["environment"]["setup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            torch.cuda.empty_cache()
            result["environment"]["timing_resident_allocated_bytes"] = torch.cuda.memory_allocated(device)
        for group in range(measure["groups"]):
            order = list(live)
            rng.shuffle(order)
            for position, name in enumerate(order):
                state = live[name]
                if state["record"]["status"] != "ok":
                    continue
                try:
                    calls = state["calls"]
                    sync()
                    if cuda:
                        torch.cuda.reset_peak_memory_stats(device)
                        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    if cuda:
                        start_event.record()
                    for invocation in range(measure["calls_per_group"]):
                        calls[(group * measure["calls_per_group"] + invocation) % len(calls)]()
                    if cuda:
                        end_event.record()
                    sync()
                    wall_ms = (time.perf_counter() - wall_start) * 1000 / measure["calls_per_group"]
                    event_ms = start_event.elapsed_time(end_event) / measure["calls_per_group"] if cuda else None
                    result["samples"].append({"case_id": case["id"], "backend": name,
                        "mode": payload["mode"], "process_index": payload["process_index"],
                        "group": group, "paired_order": order, "order_position": position,
                        "calls": measure["calls_per_group"], "event_ms": event_ms, "wall_ms": wall_ms,
                        "sample_unit": "group_mean",
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
                        "iterations": [getattr(app, "last_iterations", None) for app in state["apps"]] if dynamic else None})
                except Exception as exc:
                    state["record"].update(status="failed", implementation_pass=False,
                                             error=str(exc), traceback=traceback.format_exc())
        for name, state in live.items():
            state["record"].update(_time_summary([s for s in result["samples"] if s["backend"] == name]))
            if "nominal_flops" in state["record"]["application"]:
                flops = state["record"]["application"]["nominal_flops"]
                elapsed = state["record"].get("steady_wall_ms")
                state["record"].update(nominal_flops=flops,
                    effective_gflops=flops / (elapsed * 1e6) if elapsed and elapsed > 0 else None,
                    throughput_time_basis="steady_wall_ms")
            if case_kind(case) == "linpack":
                record = state["record"]
                flops = record["application"]["nominal_flops"]
                record["nominal_flops"] = flops
                elapsed = record.get("steady_wall_ms")
                record["effective_gflops"] = flops / (elapsed * 1e6) if elapsed and elapsed > 0 else None
                record["throughput_time_basis"] = "steady_wall_ms"
                record["throughput_convention"] = "equivalent classical FP32 solve; not FP64 HPL Rmax"
            try:
                for index, app in enumerate(state["apps"]):
                    sync()
                    values = app.validate(state["outputs"][index], app.make_reference())
                    state["record"]["application_quality_pass"] &= values["quality_pass"] is True
                    result["metrics"].append({"case_id": case["id"], "backend": name, "mode": payload["mode"],
                        "process_index": payload["process_index"], "phase": "timed_path_after", "input_slot": index,
                        "input": app.metadata(), "values": values})
                state["record"]["backend_metadata"] = state["backend"].metadata()
                state["backend"].close()
            except Exception as exc:
                state["record"].update(status="failed", implementation_pass=False, error=str(exc), traceback=traceback.format_exc())
    _finalize_pairs(result, payload["baseline"])
    return result


def _bool_csv(value):
    return True if str(value).lower() in {"1", "true"} else False if str(value).lower() in {"0", "false"} else None


def _float_csv(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def gemm_worker(payload):
    case, measure = payload["case"], payload["measurement"]
    binary = payload.get("gemm_binary")
    if payload["device"] == "cpu":
        return {"runs": [_mark_failure(payload, name, NotImplementedError("native GEMM binary requires CUDA"), "unsupported")
                         for name in payload["backends"]], "samples": [], "metrics": []}
    if not binary or not Path(binary).is_file():
        raise FileNotFoundError("--gemm-binary must point to a built megafmm_bench executable")
    folder = Path(payload["artifact_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "gemm.csv"
    args = [binary, "--shapes", f"{case['m']},{case['n']},{case['k']}", "--dtype", case.get("dtype", "fp32"),
            "--algorithm", case.get("algorithm", "sw-tuned"), "--trials", str(measure["calls_per_group"]),
            "--warmup", str(measure["warmup"]), "--groups", str(measure["groups"]),
            "--correct-trials", str(measure["fresh_quality_trials"]), "--rotate-inputs", str(measure["rotate_inputs"]),
            "--output", str(csv_path)]
    args += ["--device", payload["device"].partition(":")[2] or "0"]
    for option, flag in (("workspace_limit", "--workspace-limit"), ("heuristic_candidates", "--heuristic-candidates"),
                         ("tuning_input_sets", "--tuning-input-sets"), ("sw_tuning_candidates", "--sw-tuning-candidates")):
        if option in payload["backend_options"]:
            args += [flag, str(payload["backend_options"][option])]
    if payload["mode"] == "graph":
        args.append("--graph")
    completed = subprocess.run(args, capture_output=True, text=True, timeout=measure["timeout_seconds"])
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    write_json(folder / "command.json", args)
    if not csv_path.exists():
        raise RuntimeError(f"GEMM process exited {completed.returncode}, no CSV; see {folder}")
    with csv_path.open() as csv_stream:
        rows = list(csv.DictReader(csv_stream))
    if len(rows) != 1:
        raise RuntimeError(f"expected one GEMM CSV row, got {len(rows)}")
    row = rows[0]
    result = {"runs": [], "samples": [], "metrics": []}
    for name, prefix in [("cublaslt", "dense"), ("megafmm", "solution")]:
        # megafmm_bench: 0=completed/pass, 2=completed/numerical rejection,
        # 1=caught execution exception. A completed measurement can fail quality.
        execution_completed = completed.returncode in {0, 2}
        run = _base_record(payload, name, status="ok" if execution_completed else "failed",
            implementation_pass=execution_completed and _float_csv(row, "nonfinite") == 0,
            gemm_gate_pass=_bool_csv(row.get("numerical_pass")) if name == "megafmm" else None,
            native_accepted=_bool_csv(row.get("accepted")), native_precision_required=_bool_csv(row.get("precision_required")),
            native_process_returncode=completed.returncode, raw_csv=str(csv_path),
            candidates_csv=str(csv_path) + ".sw_tuning.csv", raw_row=row,
            steady_event_ms=_float_csv(row, prefix + "_event_ms"),
            steady_wall_ms=_float_csv(row, prefix + "_wall_ms"),
            setup_ms=None, tune_ms=None, capture_ms=None, cold_setup_ms=None,
            shared_setup_ms=_float_csv(row, "setup_ms"), shared_tune_ms=_float_csv(row, "tune_ms"),
            shared_capture_ms=_float_csv(row, "graph_setup_ms"),
            setup_cost_note="legacy GEMM binary exposes costs jointly for both backends; cannot attribute",
            actual_route="Dense" if name == "cublaslt" else row.get("actual"),
            input_hash=None, input_hash_note="legacy GEMM generator uses fixed internal seeds; no input hash export",
            input_seed_control="legacy internal fixed seeds; process repeat changes scheduling, not generated input",
            measurement=measure, quantile_unit=row.get("quantile_unit"),
            resolved_policy=payload["backend_options"], requested_device=payload["device"],
            scope="synthetic_gemm", binary_sha256=sha256(binary))
        result["runs"].append(run)
        result["metrics"].append({"case_id": case["id"], "backend": name, "mode": payload["mode"],
            "process_index": payload["process_index"], "values": {"relative_frobenius": _float_csv(row, prefix + "_ef"),
            "max_absolute": _float_csv(row, prefix + "_maxabs"), "gemm_gate_pass": run["gemm_gate_pass"],
            "native_accepted": run["native_accepted"]}})
    timing = re.compile(r"TIME .*?group=(\d+) case=(dense|solution) event_ms=([\deE.+-]+) wall_ms=([\deE.+-]+)")
    for match in timing.finditer(completed.stderr):
        result["samples"].append({"case_id": case["id"], "backend": "cublaslt" if match[2] == "dense" else "megafmm",
            "mode": payload["mode"], "process_index": payload["process_index"], "group": int(match[1]),
            "event_ms": float(match[3]), "wall_ms": float(match[4]), "calls": measure["calls_per_group"],
            "sample_unit": "group_mean", "order_position": len(result["samples"]) % 2})
    _finalize_pairs(result, payload["baseline"])
    return result


def worker(payload):
    kind = case_kind(payload["case"])
    try:
        return gemm_worker(payload) if kind == "gemm" else application_worker(payload)
    except Exception as exc:
        return {"runs": [_mark_failure(payload, name, exc) for name in payload["backends"]], "samples": [], "metrics": []}


def _append_jsonl(path, rows):
    with Path(path).open("a") as stream:
        for row in rows:
            stream.write(json.dumps(_json(row), sort_keys=True, allow_nan=False) + "\n")


def rebuild_collections(out):
    """Worker results are authoritative: recover from interruption before parent append."""
    collected = {name: [] for name in ("runs", "samples", "metrics")}
    for result_path in sorted((Path(out) / "workers").glob("*/result.json")):
        data = json.loads(result_path.read_text())
        for name in collected:
            for row in data[name]:
                row["worker_key"] = result_path.parent.name
                if name == "runs":
                    row["process_environment"] = data.get("environment", {})
                collected[name].append(row)
    for name, rows in collected.items():
        temporary = Path(out) / f"{name}.jsonl.tmp"
        temporary.write_text("".join(json.dumps(_json(row), sort_keys=True, allow_nan=False) + "\n" for row in rows))
        temporary.replace(Path(out) / f"{name}.jsonl")
    return collected


def _execute_process(command, timeout):
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        # The worker can own a native GEMM subprocess; terminate the whole process group.
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
        exc.stdout, exc.stderr = stdout, stderr
        raise


def _quality_failed(run):
    if run.get("status") != "ok":
        return False
    if run.get("kind") == "gemm":
        return run.get("backend") == "megafmm" and run.get("gemm_gate_pass") is False
    return run.get("application_quality_pass") is False


def run_suite(config, args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolved_path = out / "resolved_config.json"
    if resolved_path.exists():
        if not args.resume:
            raise ConfigurationError(f"{out} already contains a suite; use --resume or another --output")
        if json.loads(resolved_path.read_text()) != config:
            raise ConfigurationError("--resume configuration differs from saved resolved_config.json")
        saved_manifest = json.loads((out / "manifest.json").read_text())
        if saved_manifest.get("device") != args.device:
            raise ConfigurationError("--resume device differs from original execution")
        if saved_manifest.get("source_hashes") != source_hashes():
            raise ConfigurationError("--resume source hashes differ from original execution")
        if saved_manifest.get("input_asset_hashes") != input_hashes(config):
            raise ConfigurationError("--resume external input hashes differ from original execution")
        for option, recorded in ((args.library, "library_sha256"), (args.gemm_binary, "gemm_binary_sha256")):
            if (sha256(option) if option else None) != saved_manifest.get(recorded):
                raise ConfigurationError("--resume binary/library hash differs from original execution")
        if saved_manifest.get("runtime_fingerprint") is not None and saved_manifest["runtime_fingerprint"] != runtime_fingerprint(args.device):
            raise ConfigurationError("--resume physical GPU or Python/PyTorch/CUDA runtime differs from original execution")
    else:
        identity = manifest(config, args)
        identity["runtime_fingerprint"] = runtime_fingerprint(args.device)
        write_json(resolved_path, config)
        write_json(out / "manifest.json", identity)
    failures = 0
    tasks = []
    for case in config["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        for mode in case.get("modes", config["modes"]):
            for process_index in range(config["measurement"]["independent_processes"]):
                tasks.append((case, mode, process_index))
    if not tasks:
        raise ConfigurationError("no cases selected")
    rng = random.Random(config["measurement"]["seed"])
    rng.shuffle(tasks)
    for case, mode, process_index in tasks:
        key = f"{case['id']}__{mode}__p{process_index}"
        folder = out / "workers" / key
        result_path = folder / "result.json"
        if args.resume and result_path.exists():
            old = json.loads(result_path.read_text())
            # Terminal statuses are evidence, not permission to silently erase failures.
            failures += sum(run.get("status") not in {"ok", "skipped"} for run in old["runs"])
            continue
        folder.mkdir(parents=True, exist_ok=True)
        case_seed = int(hashlib.sha256(case["id"].encode()).hexdigest()[:8], 16)
        payload = {"case": case, "mode": mode, "process_index": process_index,
                   "seed": config["measurement"]["seed"] + case_seed + process_index * 1000000,
                   "measurement": config["measurement"], "backends": case.get("backends", config["backends"]),
                   "baseline": config["baseline"], "backend_options": config["backend_options"],
                   "library": str(Path(args.library).resolve()) if args.library else None,
                   "gemm_binary": str(Path(args.gemm_binary).resolve()) if args.gemm_binary else None,
                   "device": args.device, "artifact_dir": str(folder / "artifacts")}
        write_json(folder / "payload.json", payload)
        command = [sys.executable, str(ROOT / "scripts" / "run_suite.py"), "--worker", str(folder / "payload.json"),
                   "--worker-output", str(result_path)]
        print(f"[{key}] start", flush=True)
        try:
            proc = _execute_process(command, config["measurement"]["timeout_seconds"])
            (folder / "stdout.log").write_text(proc.stdout)
            (folder / "stderr.log").write_text(proc.stderr)
            if not result_path.exists():
                raise RuntimeError(f"worker exited {proc.returncode} without result; see {folder / 'stderr.log'}")
            data = json.loads(result_path.read_text())
        except (subprocess.TimeoutExpired, OSError, RuntimeError, json.JSONDecodeError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                (folder / "stdout.log").write_text(exc.stdout or "")
                (folder / "stderr.log").write_text(exc.stderr or "")
            data = {"runs": [_mark_failure(payload, name, exc,
                "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed") for name in payload["backends"]],
                "samples": [], "metrics": []}
            write_json(result_path, data)
        for collection in ("runs", "samples", "metrics"):
            for row in data[collection]:
                row["worker_key"] = key
            _append_jsonl(out / f"{collection}.jsonl", data[collection])
        failures += sum(run.get("status") not in {"ok", "skipped"} for run in data["runs"])
        print(f"[{key}] " + ", ".join(f"{run['backend']}={run['status']}" for run in data["runs"]), flush=True)
    from bench.suite.report import build_report
    collections = rebuild_collections(out)
    failures = sum(run.get("status") not in {"ok", "skipped"} or
                   (run.get("status") == "ok" and run.get("implementation_pass") is False) for run in collections["runs"])
    quality_failures = sum(_quality_failed(run) for run in collections["runs"])
    build_report(out)
    print(f"Results: {out / 'report.html'}", flush=True)
    return 1 if failures else 2 if quality_failures and not args.allow_quality_failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "bench" / "configs" / "smoke.json"), help="versioned JSON config; relative input_path is resolved against the repository root")
    p.add_argument("--output", default=str(ROOT / "results" / "suites" / time.strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--library", help="native megafmm bridge shared library")
    p.add_argument("--gemm-binary", help="existing megafmm_bench executable")
    p.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    p.add_argument("--list", action="store_true", help="validate and list config without importing torch")
    p.add_argument("--dry-run", action="store_true", help="print resolved plan without importing torch or launching workers")
    p.add_argument("--case", action="append", help="select exact case id (repeatable)")
    p.add_argument("--resume", action="store_true", help="continue missing worker tasks; retain terminal failures")
    p.add_argument("--allow-quality-failures", action="store_true", help="diagnostic run: retain failed numerical gates without exit code 2")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--worker-output", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        payload = json.loads(Path(args.worker).read_text())
        result = worker(payload)
        write_json(args.worker_output, result)
        return 1 if any(r["status"] not in {"ok", "skipped"} for r in result["runs"]) else 0
    try:
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
            raise ConfigurationError("device must be cpu, cuda, or cuda:N")
        config = load_config(args.config)
        args.library = resolved_library(args.library)
        if args.case and set(args.case) - {c["id"] for c in config["cases"]}:
            raise ConfigurationError("--case contains an unknown case id")
        selected = [c for c in config["cases"] if not args.case or c["id"] in args.case]
        if args.list:
            for case in selected:
                print(f"{case['id']}\t{case.get('app', case.get('application', case.get('kind')))}\t{case.get('dtype', 'fp32')}\t{case.get('input_scope', 'synthetic_algorithm')}")
            return 0
        if args.dry_run:
            print(json.dumps({"config": config, "selected_case_ids": [c["id"] for c in selected],
                "worker_processes": sum(len(c.get('modes', config['modes'])) for c in selected) * config['measurement']['independent_processes'],
                "device": args.device, "output": str(Path(args.output).resolve())}, indent=2))
            return 0
        return run_suite(config, args)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
