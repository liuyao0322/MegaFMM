"""Local-only causal-LM quality and latency evaluation with replaceable GEMMs.

Importing this module needs only the Python standard library.  Optional torch,
NumPy and Transformers imports happen after command-line parsing.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time


DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint):
    """Hash model/config/tokenizer files, including all weight shards."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--checkpoint must be an existing local directory")
    patterns = ("*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py")
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and any(fnmatch.fnmatch(p.name, pattern) for pattern in patterns))
    if not files:
        raise ValueError(f"No model files found in {root}")
    entries = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                "sha256": sha256_file(p)} for p in files]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"path": str(root), "sha256": hashlib.sha256(canonical).hexdigest(),
            "files": entries}


def load_token_arrays(path):
    """Load integer [batch, sequence] tokens without pickle or arbitrary code.

    NPZ: input_ids, optional attention_mask. PT: tensor, flat tensor dict, or
    {'tensors': tensor_dict, 'metadata': JSON-compatible dict}; weights_only=True.
    A single sequence may be stored as rank one and is made into one batch.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            if "input_ids" not in archive:
                raise ValueError("Token NPZ must contain input_ids")
            arrays = {key: archive[key].copy() for key in ("input_ids", "attention_mask")
                      if key in archive}
    elif path.suffix.lower() == ".pt":
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, torch.Tensor):
            raw = {"input_ids": raw}
        if isinstance(raw, dict) and "tensors" in raw:
            raw = raw["tensors"]
        if not isinstance(raw, dict) or "input_ids" not in raw:
            raise ValueError("Token PT must be a tensor or contain input_ids")
        arrays = {}
        for key in ("input_ids", "attention_mask"):
            if key in raw:
                if not isinstance(raw[key], torch.Tensor):
                    raise ValueError(f"{key} must be a tensor")
                arrays[key] = raw[key].detach().cpu().numpy()
    else:
        raise ValueError("Tokens must use .npz or .pt")
    import numpy as np
    ids = arrays["input_ids"]
    if not np.issubdtype(ids.dtype, np.integer) or ids.dtype == np.bool_:
        raise ValueError("input_ids must contain integers")
    if ids.ndim == 1:
        ids = ids[None, :]
    if ids.ndim != 2 or not ids.shape[0] or ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, sequence>=2]")
    if np.any(ids < 0):
        raise ValueError("input_ids cannot be negative")
    if np.any(ids > np.iinfo(np.int64).max):
        raise ValueError("input_ids must fit signed 64-bit integers")
    arrays["input_ids"] = ids.astype(np.int64, copy=False)
    if "attention_mask" in arrays:
        mask = arrays["attention_mask"]
        if mask.ndim == 1:
            mask = mask[None, :]
        if mask.shape != ids.shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must match input_ids and contain 0/1")
        if not np.any(mask[:, 1:] * mask[:, :-1]):
            raise ValueError("attention_mask contains no valid next-token targets")
        arrays["attention_mask"] = mask.astype(np.int64, copy=False)
    return arrays


def token_tensors(path, device):
    import torch
    return {key: torch.as_tensor(value, device=device, dtype=torch.long)
            for key, value in load_token_arrays(path).items()}


def load_local_model(checkpoint, dtype, device, attention="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Checkpoint must be an existing local directory")
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype}")
    kwargs = {"local_files_only": True, "trust_remote_code": False,
              "torch_dtype": getattr(torch, DTYPES[dtype])}
    if attention != "default":
        kwargs["attn_implementation"] = attention
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def select_linear_names(model, patterns=None, scope="mlp"):
    import torch
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or (not patterns and "lm_head" in name):
            continue
        if patterns:
            selected = any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        elif scope == "mlp":
            selected = any(part in {"mlp", "feed_forward", "ffn"}
                           for part in name.split("."))
        else:
            selected = ("mlp" in name or "self_attn" in name
                        or "attention" in name or "feed_forward" in name)
        if selected:
            names.append(name)
    if not names:
        raise ValueError("No nn.Linear modules match the requested scope/patterns")
    return names


def _parent_and_leaf(model, name):
    parent_name, _, leaf = name.rpartition(".")
    return model.get_submodule(parent_name) if parent_name else model, leaf


@contextlib.contextmanager
def replace_linears(model, names, backend):
    """Replace only chosen linears; preserve attention and restore on all exits."""
    import torch

    class BackendLinear(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight = original.weight
            self.bias = original.bias
            self.in_features = original.in_features
            self.out_features = original.out_features

        def forward(self, inputs):
            shape = inputs.shape[:-1] + (self.out_features,)
            output = backend.gemm(inputs.reshape(-1, self.in_features), self.weight.T)
            # Both native backends produce FP32. Bias and the required cast are
            # part of the measured model, with the same contract on both paths.
            if self.bias is not None:
                output = output + self.bias.float()
            return output.to(inputs.dtype).reshape(shape)

    originals = []
    try:
        for name in names:
            parent, leaf = _parent_and_leaf(model, name)
            original = getattr(parent, leaf)
            originals.append((parent, leaf, original))
            setattr(parent, leaf, BackendLinear(original))
        yield
    finally:
        for parent, leaf, original in reversed(originals):
            setattr(parent, leaf, original)


def causal_quality(logits, batch, reference=None):
    import torch
    predictions = logits[:, :-1].float()
    targets = batch["input_ids"][:, 1:]
    valid = torch.ones_like(targets, dtype=torch.bool)
    if "attention_mask" in batch:
        mask = batch["attention_mask"]
        valid = mask[:, 1:].bool() & mask[:, :-1].bool()
    count = int(valid.sum().item())
    if not count:
        raise ValueError("No next-token targets to evaluate")
    losses = torch.nn.functional.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1),
        reduction="none").reshape(targets.shape)
    nll = float(losses[valid].double().mean().item())
    metrics = {"nll": nll, "perplexity": math.exp(nll) if nll < 709 else None,
               "target_tokens": count, "finite": bool(torch.isfinite(logits).all().item())}
    if reference is not None:
        ref = reference.to(device=logits.device, dtype=torch.float32)
        diff = logits.float() - ref
        metrics.update({"logits_relative_l2": float((torch.linalg.vector_norm(diff.double()) /
                        torch.linalg.vector_norm(ref.double()).clamp_min(1e-30)).item()),
                        "logits_max_abs": float(diff.abs().max().item())})
    return metrics


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def measure_forward(model, batch, device, warmup, repeats):
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            model(**batch, use_cache=False)
        _sync(device)
        samples = []
        for index in range(repeats):
            if str(device).startswith("cuda"):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                output = model(**batch, use_cache=False)
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                event_ms = float(start.elapsed_time(end))
            else:
                wall_start = time.perf_counter()
                output = model(**batch, use_cache=False)
                wall_ms = (time.perf_counter() - wall_start) * 1000
                event_ms = None
            del output
            samples.append({"index": index, "event_ms": event_ms, "wall_ms": wall_ms})
    return {"median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
            "median_event_ms": statistics.median(s["event_ms"] for s in samples)
            if samples[0]["event_ms"] is not None else None, "samples": samples}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Existing local Hugging Face checkpoint directory")
    parser.add_argument("--tokens", required=True, help="Local NPZ input_ids/attention_mask or safe PT tokens")
    parser.add_argument("--tuning-tokens", help="Optional separate local tuning fixture with the same token shape")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("sdpa", "eager", "default"), default="sdpa")
    parser.add_argument("--scope", choices=("mlp", "projections"), default="mlp")
    parser.add_argument("--modules", nargs="+", help="Exact names or shell-style patterns for nn.Linear modules")
    parser.add_argument("--backends", nargs="+", choices=("torch", "cublaslt", "megafmm", "megafmm_fixed"),
                        default=["torch", "cublaslt", "megafmm"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nll-delta", type=float, default=0.01,
                        help="Predeclared absolute NLL deviation gate against original torch model")
    parser.add_argument("--max-logits-relative-l2", type=float, default=0.05)
    parser.add_argument("--backend-options", default="{}", help="JSON options shared by native Backend constructors")
    parser.add_argument("--allow-quality-failures", action="store_true",
                        help="Diagnostic mode: still record failed quality gates but return zero if execution succeeds")
    return parser


def evaluate(args):
    import torch
    from .backends import Backend
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be >=0 and repeats >=1")
    if any(not math.isfinite(value) or value < 0
           for value in (args.max_nll_delta, args.max_logits_relative_l2)):
        raise ValueError("Quality thresholds must be finite and nonnegative")
    options = json.loads(args.backend_options)
    if not isinstance(options, dict):
        raise ValueError("--backend-options must decode to an object")
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = checkpoint_manifest(args.checkpoint)
    batch = token_tensors(args.tokens, args.device)
    tuning_batch = token_tensors(args.tuning_tokens, args.device) if args.tuning_tokens else batch
    if tuning_batch["input_ids"].shape != batch["input_ids"].shape:
        raise ValueError("Tuning and evaluation tokens must have the same [batch, sequence] shape")
    start = time.perf_counter()
    model = load_local_model(args.checkpoint, args.dtype, args.device, args.attention)
    _sync(args.device)
    load_ms = (time.perf_counter() - start) * 1000
    names = select_linear_names(model, args.modules, args.scope)
    with torch.no_grad():
        start = time.perf_counter()
        ref_logits = model(**batch, use_cache=False).logits
        _sync(args.device)
        torch_cold_ms = (time.perf_counter() - start) * 1000
        reference_metrics = causal_quality(ref_logits, batch)
        reference = ref_logits.detach().float().cpu()
        del ref_logits
    report = {"schema_version": 1, "experiment": "causal_lm_prefill", "status": "running",
              "checkpoint": checkpoint, "tokens": {"path": str(Path(args.tokens).resolve()),
              "sha256": sha256_file(args.tokens), "shape": list(batch["input_ids"].shape)},
              "dtype": args.dtype, "attention_requested": args.attention,
              "attention_actual": getattr(model.config, "_attn_implementation", None),
              "seed": args.seed, "modules": names, "model_load_ms": load_ms,
              "tuning_tokens": {"path": str(Path(args.tuning_tokens or args.tokens).resolve()),
                                "sha256": sha256_file(args.tuning_tokens or args.tokens),
                                "separate_file": (Path(args.tuning_tokens or args.tokens).resolve()
                                                  != Path(args.tokens).resolve()),
                                "different_input_ids": not torch.equal(tuning_batch["input_ids"], batch["input_ids"])},
              "quality_contract": {"max_nll_delta": args.max_nll_delta,
                                   "max_logits_relative_l2": args.max_logits_relative_l2},
              "reference": reference_metrics, "backends": [],
              "scope_note": "Fixed local token fixture; no training or dataset-general quality claim."}
    backend_order = list(args.backends)
    random.Random(args.seed).shuffle(backend_order)
    report["backend_measurement_order"] = backend_order
    for name in backend_order:
        backend = None
        try:
            constructor_start = time.perf_counter()
            if name != "torch":
                backend = Backend(name, args.device, args.dtype, options=options)
            constructor_ms = (time.perf_counter() - constructor_start) * 1000
            context = replace_linears(model, names, backend) if backend else contextlib.nullcontext()
            with context, torch.no_grad():
                # First forward makes all plans and performs tuning. Freeze
                # prevents accidental tuning during quality and timed forwards.
                cold_start = time.perf_counter()
                cold_logits = model(**tuning_batch, use_cache=False).logits
                _sync(args.device)
                dry_run_ms = (time.perf_counter() - cold_start) * 1000
                del cold_logits
                if backend:
                    backend.freeze()
                logits = model(**batch, use_cache=False).logits
                metrics = causal_quality(logits, batch, reference)
                del logits
                nll_delta = metrics["nll"] - reference_metrics["nll"]
                metrics["nll_delta"] = nll_delta
                quality_pass = (metrics["finite"] and math.isfinite(metrics["nll"])
                                and abs(nll_delta) <= args.max_nll_delta
                                and metrics["logits_relative_l2"] <= args.max_logits_relative_l2)
                if backend:
                    backend.record_inputs = False
                measurement = measure_forward(model, batch, args.device, args.warmup, args.repeats)
                report["backends"].append({"backend": name, "status": "completed",
                    "implementation_pass": True, "application_quality_pass": quality_pass,
                    "gemm_gate_pass": None, "metrics": metrics, "timing": measurement,
                    "setup": {"constructor_ms": constructor_ms, "dry_run_with_tuning_ms": dry_run_ms,
                              "torch_reference_first_forward_ms": torch_cold_ms},
                    "metadata": backend.metadata() if backend else {
                        "route": "unmodified_torch_nn_Linear", "dtype": args.dtype,
                        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)}})
        except Exception as error:
            failure = {"backend": name, "status": "failed",
                "implementation_pass": False, "application_quality_pass": False,
                "gemm_gate_pass": None, "error": f"{type(error).__name__}: {error}"}
            if backend:
                with contextlib.suppress(Exception):
                    failure["metadata"] = backend.metadata()
            report["backends"].append(failure)
        finally:
            if backend:
                backend.close()
    completed = {entry["backend"]: entry for entry in report["backends"] if entry["status"] == "completed"}
    baseline = completed.get("cublaslt") or completed.get("torch")
    for entry in report["backends"]:
        if entry["status"] == "completed" and baseline:
            entry["baseline"] = baseline["backend"]
            entry["baseline_application_quality_pass"] = baseline["application_quality_pass"]
            entry["speedup"] = baseline["timing"]["median_wall_ms"] / entry["timing"]["median_wall_ms"]
            entry["performance_win"] = entry["speedup"] > 1
            entry["qualified_win"] = (entry["performance_win"] and entry["application_quality_pass"]
                                      and baseline["application_quality_pass"])
    report["status"] = "completed" if all(e["status"] == "completed" for e in report["backends"]) else "partial_failure"
    report["application_quality_pass"] = (report["status"] == "completed"
                                          and all(e["application_quality_pass"] for e in report["backends"]))
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"model evaluation failed: {error}\n")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False ensures nonfinite failures cannot masquerade as JSON numbers.
    output.write_text(json.dumps(_json_finite(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "status": report["status"],
                      "application_quality_pass": report["application_quality_pass"],
                      "backends": [{"backend": row["backend"], "status": row["status"],
                                    "application_quality_pass": row["application_quality_pass"]}
                                   for row in report["backends"]]}))
    if report["status"] != "completed":
        return 1
    return 0 if report["application_quality_pass"] or args.allow_quality_failures else 2


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value
