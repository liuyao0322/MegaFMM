#!/usr/bin/env python3
"""Audit scientific-suite coverage and provenance without importing GPU libraries.

Exit 1 denotes incomplete or inconsistent evidence. Numerical rejection alone
does not fail integrity: inspect execution_pass and quality_all_pass separately.
Only audit.json is written; worker records and aggregate collections are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


NATIVE = {"cublaslt", "megafmm", "megafmm_fixed"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def identity(row):
    return row["case_id"], row["mode"], row["process_index"], row["backend"]


def worker_key(key):
    return f"{key[0]}__{key[1]}__p{key[2]}"


def quality(row):
    if row.get("kind") == "gemm":
        return row.get("implementation_pass") if row.get("backend") == "cublaslt" else row.get("gemm_gate_pass")
    return row.get("application_quality_pass")


def canonical(row):
    return json.dumps({k: v for k, v in row.items() if k not in {"worker_key", "process_environment"}},
                      sort_keys=True, allow_nan=False)


def exact_canonical(row):
    """Retain all metric evidence, including JSON-safe nonfinite strings."""
    return json.dumps(row, sort_keys=True, allow_nan=False)


def audit(directory, source_root=None, build_manifest_path=None, relative_tolerance=1e-9):
    directory = Path(directory).resolve()
    issues, notes, checks = [], [], {}
    counts = Counter({name: 0 for name in (
        "expected_workers", "observed_worker_results", "missing_workers", "unexpected_workers",
        "duplicate_worker_identities", "reused_worker_pids", "expected_runs", "observed_runs",
        "missing_runs", "unexpected_runs", "duplicate_runs", "aggregate_runs", "aggregate_samples",
        "observed_sample_groups", "expected_successful_sample_groups", "missing_sample_groups",
        "duplicate_sample_groups", "unexpected_sample_groups", "successful_runs", "execution_failed_runs",
        "quality_failed_runs", "quality_unchecked_runs", "checked_source_files", "checked_hashed_files",
        "observed_metrics", "aggregate_metrics", "missing_aggregate_metrics", "unexpected_aggregate_metrics",
        "duplicate_aggregate_metrics", "expected_quality_observations", "observed_quality_observations",
        "missing_quality_observations", "unexpected_quality_observations", "duplicate_quality_observations",
        "invalid_quality_observations", "quality_aggregation_mismatches")})

    def reject(code, detail, key=None):
        item = {"code": code, "detail": str(detail)}
        if key is not None:
            item["identity"] = list(key) if isinstance(key, tuple) else str(key)
        issues.append(item)

    def read_json(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            reject("unreadable_json", f"{path}: {exc}")
            return None

    def read_jsonl(path):
        try:
            lines = Path(path).read_text().splitlines()
        except OSError as exc:
            reject("missing_collection", f"{path}: {exc}")
            return []
        rows = []
        for number, line in enumerate(lines, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    rows.append(row)
                except ValueError as exc:
                    reject("invalid_collection_json", f"{path}:{number}: {exc}")
        return rows

    config, manifest = read_json(directory / "resolved_config.json"), read_json(directory / "manifest.json")
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        return {"schema_version": 1, "directory": str(directory), "integrity_pass": False,
                "complete": False, "execution_pass": False, "quality_all_pass": False,
                "issues": issues, "counts": {}, "checks": checks}
    cases = {c["id"]: c for c in config["cases"]}
    if len(cases) != len(config["cases"]):
        reject("duplicate_case_id", "resolved config contains repeated case IDs")
    selected = manifest.get("selected_case_ids", list(cases))
    if "selected_case_ids" not in manifest:
        notes.append("Manifest has no selected_case_ids; all configured cases are expected.")
    if len(selected) != len(set(selected)) or set(selected) - set(cases):
        reject("invalid_selected_cases", selected)
    measure = config["measurement"]
    expected_workers, expected_runs = {}, set()
    for name in selected:
        if name not in cases:
            continue
        case = cases[name]
        for mode in case.get("modes", config["modes"]):
            for process in range(measure["independent_processes"]):
                key = name, mode, process
                expected_workers[key] = case.get("backends", config["backends"])
                expected_runs.update((*key, backend) for backend in expected_workers[key])
    counts["expected_workers"] = len(expected_workers)
    counts["expected_runs"] = len(expected_runs)
    expected_names = {worker_key(key): key for key in expected_workers}
    result_paths = sorted((directory / "workers").glob("*/result.json"))
    counts["observed_worker_results"] = len(result_paths)
    present_names = {p.parent.name for p in result_paths}
    for name in sorted(set(expected_names) - present_names):
        counts["missing_workers"] += 1
        reject("missing_worker", name)
    runs, samples, metrics, worker_ids = [], [], [], defaultdict(list)
    pid_owners = defaultdict(set)
    for path in result_paths:
        folder = path.parent
        data = read_json(path)
        payload = read_json(folder / "payload.json")
        if folder.name not in expected_names:
            counts["unexpected_workers"] += 1
            reject("unexpected_worker", folder.name)
        if not isinstance(data, dict) or not isinstance(payload, dict):
            continue
        key = payload["case"]["id"], payload["mode"], payload["process_index"]
        worker_ids[key].append(folder.name)
        if folder.name != worker_key(key):
            reject("worker_identity_mismatch", folder.name, key)
        if key in expected_workers and payload.get("backends") != expected_workers[key]:
            reject("payload_backend_mismatch", payload.get("backends"), key)
        if key in expected_workers:
            expected_seed = measure["seed"] + int(hashlib.sha256(key[0].encode()).hexdigest()[:8], 16) + key[2] * 1000000
            if payload.get("seed") != expected_seed:
                reject("payload_seed_mismatch", {"actual": payload.get("seed"), "expected": expected_seed}, key)
        worker_runs, worker_samples = data.get("runs", []), data.get("samples", [])
        pids = set()
        for row in worker_runs:
            try:
                row_key = identity(row)
            except (KeyError, TypeError) as exc:
                reject("malformed_run", f"{path}: {exc}")
                continue
            if row_key[:3] != key:
                reject("run_worker_identity_mismatch", folder.name, row_key)
            if row.get("seed") != payload.get("seed"):
                reject("run_seed_mismatch", row.get("seed"), row_key)
            pid = row.get("worker_pid")
            if type(pid) is not int or pid <= 0:
                reject("invalid_worker_pid", pid, row_key)
            else:
                pids.add(pid)
                pid_owners[pid].add(key)
            runs.append(row)
        if len(pids) != 1:
            reject("worker_pid_disagreement", sorted(pids), key)
        for row in worker_samples:
            try:
                row_key = identity(row)
            except (KeyError, TypeError) as exc:
                reject("malformed_sample", f"{path}: {exc}")
                continue
            if row_key[:3] != key:
                reject("sample_worker_identity_mismatch", folder.name, row_key)
            samples.append(row)
        for row in data.get("metrics", []):
            try:
                row_key = identity(row)
            except (KeyError, TypeError) as exc:
                reject("malformed_metric", f"{path}: {exc}")
                continue
            if row_key[:3] != key:
                reject("metric_worker_identity_mismatch", folder.name, row_key)
            # The aggregate writer adds this one field; every other payload
            # value must survive unchanged, including nonfinite strings.
            metrics.append({**row, "worker_key": folder.name})
    for key, folders in worker_ids.items():
        if len(folders) > 1:
            counts["duplicate_worker_identities"] += len(folders) - 1
            reject("duplicate_worker_identity", folders, key)
    for pid, owners in pid_owners.items():
        if len(owners) > 1:
            counts["reused_worker_pids"] += 1
            reject("nonindependent_worker_pid", {"pid": pid, "workers": sorted(owners)})
    checks["independent_worker_pids"] = not any(i["code"] in {"invalid_worker_pid", "worker_pid_disagreement", "nonindependent_worker_pid"} for i in issues)
    by_run, by_sample, by_metric = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in runs:
        by_run[identity(row)].append(row)
    for row in samples:
        by_sample[identity(row)].append(row)
    for row in metrics:
        by_metric[identity(row)].append(row)
    counts["observed_metrics"] = len(metrics)
    counts["observed_runs"], counts["observed_sample_groups"] = len(runs), len(samples)
    for key in sorted(expected_runs - set(by_run)):
        counts["missing_runs"] += 1
        reject("missing_run", "expected backend record absent", key)
    for key, rows in by_run.items():
        if key not in expected_runs:
            counts["unexpected_runs"] += len(rows)
            reject("unexpected_run", len(rows), key)
        if len(rows) != 1:
            counts["duplicate_runs"] += len(rows) - 1
            reject("duplicate_run", len(rows), key)
    for collection, authoritative in (("runs", runs), ("samples", samples)):
        aggregate = read_jsonl(directory / f"{collection}.jsonl")
        counts[f"aggregate_{collection}"] = len(aggregate)
        try:
            if Counter(map(canonical, aggregate)) != Counter(map(canonical, authoritative)):
                reject("aggregate_mismatch", f"{collection}.jsonl differs from authoritative worker results")
        except (TypeError, ValueError) as exc:
            reject("nonstandard_record", exc)
    aggregate_metrics = read_jsonl(directory / "metrics.jsonl")
    counts["aggregate_metrics"] = len(aggregate_metrics)
    try:
        authoritative_counts = Counter(map(exact_canonical, metrics))
        aggregate_counts = Counter(map(exact_canonical, aggregate_metrics))
        missing_metrics = authoritative_counts - aggregate_counts
        extra_metrics = aggregate_counts - authoritative_counts
        counts["missing_aggregate_metrics"] = sum(missing_metrics.values())
        counts["unexpected_aggregate_metrics"] = sum(extra_metrics.values())
        counts["duplicate_aggregate_metrics"] = sum(
            max(0, count - max(1, authoritative_counts.get(record, 0)))
            for record, count in aggregate_counts.items())
        if missing_metrics or extra_metrics:
            reject("aggregate_metrics_mismatch", {
                "missing_records": counts["missing_aggregate_metrics"],
                "extra_records": counts["unexpected_aggregate_metrics"],
                "duplicate_records": counts["duplicate_aggregate_metrics"]})
    except (TypeError, ValueError) as exc:
        reject("nonstandard_metric", exc)
    for key, rows in by_metric.items():
        if key not in by_run:
            reject("orphan_metrics", len(rows), key)
    for key, rows in by_sample.items():
        if key not in by_run:
            reject("orphan_samples", len(rows), key)
        groups = Counter(r.get("group") for r in rows)
        for group, n in groups.items():
            if type(group) is not int or not 0 <= group < measure["groups"]:
                counts["unexpected_sample_groups"] += n
                reject("unexpected_sample_group", group, key)
            if n > 1:
                counts["duplicate_sample_groups"] += n - 1
                reject("duplicate_sample_group", group, key)
        for row in rows:
            if not positive(row.get("wall_ms")) or (row.get("event_ms") is not None and not positive(row["event_ms"])):
                reject("invalid_sample_time", {k: row.get(k) for k in ("group", "wall_ms", "event_ms")}, key)
            if row.get("calls") != measure["calls_per_group"]:
                reject("sample_call_count_mismatch", row.get("calls"), key)
    successful, quality_gates = [], []
    for key in sorted(expected_runs & set(by_run)):
        if len(by_run[key]) != 1:
            continue
        row = by_run[key][0]
        quality_gates.append(quality(row))
        counts["quality_failed_runs"] += quality(row) is False
        counts["quality_unchecked_runs"] += not isinstance(quality(row), bool)
        if row.get("status") != "ok" or row.get("implementation_pass") is not True:
            counts["execution_failed_runs"] += 1
            continue
        successful.append(row)
        counts["successful_runs"] += 1
        counts["expected_successful_sample_groups"] += measure["groups"]
        actual_samples = by_sample[key]
        groups = {r["group"] for r in actual_samples}
        missing = sorted(set(range(measure["groups"])) - groups)
        if missing:
            counts["missing_sample_groups"] += len(missing)
            reject("missing_sample_groups", missing, key)
        recorded_groups = row.get("group_count")
        if recorded_groups is None and row.get("kind") == "gemm":
            recorded_groups = int(row.get("raw_row", {}).get("groups", -1))
        if recorded_groups != measure["groups"]:
            reject("run_group_count_mismatch", recorded_groups, key)
        if not positive(row.get("steady_wall_ms")):
            reject("invalid_run_wall_time", row.get("steady_wall_ms"), key)
        if row.get("kind") != "gemm" and actual_samples:
            times = [s["wall_ms"] for s in actual_samples if positive(s.get("wall_ms"))]
            if times and positive(row.get("steady_wall_ms")) and not math.isclose(row["steady_wall_ms"], statistics.median(times), rel_tol=relative_tolerance):
                reject("wall_median_mismatch", {"recorded": row["steady_wall_ms"], "samples_median": statistics.median(times)}, key)
        if row.get("kind") != "gemm":
            expected_observations = {(None, trial) for trial in range(measure["fresh_quality_trials"])}
            expected_observations.update((phase, slot) for phase in ("timed_path_before", "timed_path_after")
                                         for slot in range(measure["rotate_inputs"]))
            counts["expected_quality_observations"] += len(expected_observations)
            observations = defaultdict(list)
            gates = []
            for metric in by_metric[key]:
                phase = metric.get("phase")
                if phase == "local_gemm_diagnostics":
                    continue
                index_name = "trial" if phase is None else "input_slot"
                index = metric.get(index_name)
                valid_phase = ("phase" not in metric or
                               type(phase) is str and phase in ("timed_path_before", "timed_path_after"))
                if not valid_phase or type(index) is not int or (phase, index) not in expected_observations:
                    counts["unexpected_quality_observations"] += 1
                    reject("unexpected_quality_observation", {"phase": phase, index_name: index}, key)
                    continue
                observations[(phase, index)].append(metric)
                counts["observed_quality_observations"] += 1
                values = metric.get("values")
                gate = values.get("quality_pass") if isinstance(values, dict) else None
                gates.append(gate)
                if type(gate) is not bool:
                    counts["invalid_quality_observations"] += 1
                    reject("invalid_quality_observation", {"phase": phase, index_name: index, "quality_pass": gate}, key)
            missing = expected_observations - set(observations)
            if missing:
                counts["missing_quality_observations"] += len(missing)
                reject("missing_quality_observations", sorted(missing, key=str), key)
            for observation, records in observations.items():
                if len(records) > 1:
                    counts["duplicate_quality_observations"] += len(records) - 1
                    reject("duplicate_quality_observation", {"observation": observation, "count": len(records)}, key)
            if (not missing and len(gates) == len(expected_observations)
                    and all(type(gate) is bool for gate in gates)
                    and row.get("application_quality_pass") is not all(gates)):
                counts["quality_aggregation_mismatches"] += 1
                reject("application_quality_aggregation_mismatch", {
                    "recorded": row.get("application_quality_pass"), "observations_and": all(gates)}, key)
    for key in expected_workers:
        paired = [r for r in successful if identity(r)[:3] == key]
        apps = [r for r in paired if r.get("kind") != "gemm"]
        for field in ("input_hash", "validation_seeds"):
            values = [r.get(field) for r in apps]
            if values and (any(v is None for v in values) or any(v != values[0] for v in values[1:])):
                reject("paired_input_mismatch", field, key)
        for row in apps:
            if not isinstance(row.get("input_hash"), str) or len(row["input_hash"]) != 64:
                reject("invalid_input_hash", row.get("input_hash"), identity(row))
            if len(row.get("validation_seeds", [])) != measure["fresh_quality_trials"]:
                reject("validation_trial_count_mismatch", row.get("validation_seeds"), identity(row))
            if row.get("input_sets"):
                inputs = [item.get("stored_inputs", {}) for item in row["input_sets"]]
                hashed = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
                if hashed != row.get("input_hash"):
                    reject("stored_input_hash_mismatch", hashed, identity(row))
        base_rows = by_run.get((*key, config["baseline"]), [])
        if len(base_rows) != 1:
            continue
        base = base_rows[0]
        for row in paired:
            if row["backend"] == config["baseline"] or base.get("status") != "ok" or base.get("implementation_pass") is not True:
                continue
            basis = "steady_event_ms" if row.get("kind") == "gemm" else "steady_wall_ms"
            bt, rt = base.get(basis), row.get(basis)
            if not positive(bt) or not positive(rt) or not positive(row.get("speedup")):
                reject("missing_or_invalid_speedup", row.get("speedup"), identity(row))
            elif not math.isclose(row["speedup"], bt / rt, rel_tol=relative_tolerance):
                reject("speedup_mismatch", {"recorded": row["speedup"], "recomputed": bt / rt}, identity(row))
            if row.get("speedup_time_basis") != basis:
                reject("speedup_time_basis_mismatch", row.get("speedup_time_basis"), identity(row))
    checked_files = {}

    def check_hash(path, expected_hash, context):
        path = Path(path)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            reject("missing_expected_hash", context)
            return
        if str(path) not in checked_files:
            try:
                checked_files[str(path)] = digest(path)
            except OSError as exc:
                reject("unreadable_hashed_file", f"{context}: {exc}")
                return
        if checked_files[str(path)] != expected_hash:
            reject("hash_mismatch", {"context": context, "path": str(path), "expected": expected_hash, "actual": checked_files[str(path)]})

    library = manifest.get("library")
    if library:
        library = Path(library)
        if not library.is_absolute():
            library = directory / library
        check_hash(library, manifest.get("library_sha256"), "manifest native library")
    if manifest.get("gemm_binary"):
        check_hash(manifest["gemm_binary"], manifest.get("gemm_binary_sha256"), "manifest native GEMM executable")
    for row in successful:
        if row["backend"] in NATIVE and row.get("kind") != "gemm":
            if not library or row.get("library_sha256") != manifest.get("library_sha256"):
                reject("run_library_hash_mismatch", row.get("library_sha256"), identity(row))
            loaded = row.get("backend_metadata", {}).get("library")
            if not loaded:
                reject("missing_loaded_library_path", "backend metadata library absent", identity(row))
            else:
                check_hash(loaded, row.get("library_sha256"), "worker loaded native library")
        if row.get("kind") == "gemm" and row.get("binary_sha256") != manifest.get("gemm_binary_sha256"):
            reject("run_binary_hash_mismatch", row.get("binary_sha256"), identity(row))
    embedded = manifest.get("native_build_manifest")
    build_path = Path(build_manifest_path).resolve() if build_manifest_path else (library.parent / "build_manifest.json" if library else None)
    adjacent = read_json(build_path) if build_path and build_path.is_file() else None
    if build_manifest_path and adjacent is None:
        reject("missing_build_manifest", build_path)
    if embedded is not None and adjacent is not None and embedded != adjacent:
        reject("build_manifest_mismatch", "embedded and on-disk native build manifests differ")
    build = adjacent if adjacent is not None else embedded
    checks["build_manifest"] = "checked" if build else "unavailable"
    if build:
        artifacts = build.get("artifact_sha256", {})
        if library:
            if artifacts.get(library.name) != manifest.get("library_sha256"):
                reject("build_library_hash_mismatch", library.name)
            for name, expected_hash in artifacts.items():
                check_hash(library.parent / name, expected_hash, "build artifact")
        build_sources = build.get("build_source_sha256", build.get("source_sha256", {}))
        if "native_bridge_sha256" in build:
            build_sources = {**build_sources, "bench/suite/native_bridge.cpp": build["native_bridge_sha256"]}
        for name, expected_hash in build_sources.items():
            compiled = name.startswith(("solution/", "include/")) or name == "bench/suite/native_bridge.cpp"
            if compiled and name in manifest.get("source_hashes", {}) and manifest["source_hashes"][name] != expected_hash:
                reject("build_source_hash_mismatch", name)
    if source_root:
        source_root = Path(source_root).resolve()
        for name, expected_hash in manifest.get("source_hashes", {}).items():
            check_hash(source_root / name, expected_hash, "suite source")
            counts["checked_source_files"] += 1
        if not manifest.get("source_hashes"):
            reject("missing_source_hashes", "manifest source_hashes absent or empty")
        checks["source_hashes"] = "checked"
    else:
        checks["source_hashes"] = "unchecked; pass --source-root for the frozen execution snapshot"
    counts["checked_hashed_files"] = len(checked_files)
    coverage_codes = {"missing_worker", "unexpected_worker", "duplicate_worker_identity", "missing_run", "unexpected_run",
                      "duplicate_run", "missing_sample_groups", "duplicate_sample_group", "unexpected_sample_group",
                      "missing_collection", "aggregate_mismatch", "run_group_count_mismatch", "aggregate_metrics_mismatch",
                      "missing_quality_observations", "duplicate_quality_observation", "unexpected_quality_observation"}
    checks["aggregate_metrics"] = not any(i["code"] in {"aggregate_metrics_mismatch", "nonstandard_metric", "malformed_metric"}
                                          for i in issues) and (directory / "metrics.jsonl").is_file()
    checks["application_quality_observations"] = not any(i["code"] in {
        "missing_quality_observations", "duplicate_quality_observation", "unexpected_quality_observation",
        "invalid_quality_observation", "application_quality_aggregation_mismatch", "orphan_metrics",
        "metric_worker_identity_mismatch"} for i in issues)
    complete = bool(expected_workers) and not any(i["code"] in coverage_codes for i in issues)
    execution_pass = complete and len(successful) == len(expected_runs)
    return {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "directory": str(directory), "source_root": str(source_root) if source_root else None,
            "selected_case_ids": selected, "integrity_pass": not issues, "complete": complete,
            "execution_pass": execution_pass,
            "quality_all_pass": bool(expected_runs) and len(quality_gates) == len(expected_runs)
                                and all(g is True for g in quality_gates) and checks["application_quality_observations"],
            "numerical_failure_policy": "quality failure is retained separately and does not fail integrity",
            "relative_tolerance": relative_tolerance, "counts": dict(sorted(counts.items())),
            "checks": checks, "issues": issues, "notes": notes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-root", type=Path, help="exact frozen source snapshot used for this run")
    parser.add_argument("--build-manifest", type=Path, help="optional native build manifest; otherwise use the library's adjacent manifest")
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    if not math.isfinite(args.relative_tolerance) or args.relative_tolerance < 0:
        parser.error("relative tolerance must be finite and nonnegative")
    try:
        result = audit(args.directory, args.source_root, args.build_manifest, args.relative_tolerance)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result = {"schema_version": 1, "directory": str(args.directory.resolve()), "integrity_pass": False,
                  "complete": False, "execution_pass": False, "quality_all_pass": False,
                  "issues": [{"code": "invalid_archive", "detail": str(exc)}], "counts": {}, "checks": {}}
    path = args.directory / "audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: result[k] for k in ("integrity_pass", "complete", "execution_pass", "quality_all_pass", "counts")}, sort_keys=True))
    print(path.resolve())
    return 0 if result["integrity_pass"] and result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
