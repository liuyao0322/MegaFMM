#!/usr/bin/env python3
"""Plot archived scientific-suite covariance measurements without running a GPU.

Reads completed suite artifacts and writes PNG, PDF and SVG figures. Timings and
confidence intervals come from summary.json; numerical extrema include every
saved fresh-input and timed-path validation in metrics.jsonl.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import textwrap


BACKENDS = (
    ("cublaslt", "cuBLASLt (FP32 PEDANTIC)", "#0072B2", "o"),
    ("megafmm", "MegaFMM (FP32 SW)", "#D55E00", "s"),
    ("torch", "PyTorch (strict FP32)", "#CC79A7", "^"),
    ("practical_torch", "PyTorch (TF32 allowed)", "#8C6D31", "D"),
)
FAILED_COLOR = "#E69F00"
QUALIFIED_COLOR = "#198754"
TIMING_COLOR = "#8C9198"


def finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and (value > 0 if positive else value >= 0))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def read_jsonl(path):
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def load_archive(directory):
    """Refuse a partial archive instead of plotting a favorable early subset."""
    config = read_json(directory / "resolved_config.json")
    manifest = read_json(directory / "manifest.json")
    summaries = read_json(directory / "summary.json")
    runs = read_jsonl(directory / "runs.jsonl")
    metrics = read_jsonl(directory / "metrics.jsonl")
    if not isinstance(summaries, list):
        raise ValueError("summary.json must contain the suite summary list")
    if config.get("baseline") != "cublaslt":
        raise ValueError("This figure requires cublaslt as the configured baseline")
    selected = set(manifest.get("selected_case_ids", [case["id"] for case in config["cases"]]))
    expected = set()
    for case in config["cases"]:
        if case["id"] not in selected:
            continue
        for mode in case.get("modes", config["modes"]):
            for index in range(config["measurement"]["independent_processes"]):
                for backend in case.get("backends", config["backends"]):
                    expected.add((case["id"], mode, index, backend))
    observed = set()
    for run in runs:
        identity = (run["case_id"], run["mode"], run["process_index"], run["backend"])
        if identity in observed:
            raise ValueError(f"Duplicate run identity: {identity}; rebuild the archive report first")
        observed.add(identity)
        if run.get("status") not in {"ok", "failed", "unsupported", "oom", "timeout", "skipped"}:
            raise ValueError(f"Run is not terminal: {identity}")
    missing = expected - observed
    if missing:
        raise ValueError(f"Archive is incomplete: {len(missing)} expected backend/process records are absent")
    cases = sorted((case for case in config["cases"]
                    if case["id"] in selected and case.get("kind") == "covariance" and
                    "launch" in case.get("modes", config["modes"])), key=lambda case: case["n"])
    if not cases:
        raise ValueError("No selected launch-mode covariance cases in the archive")
    if len({case["n"] for case in cases}) != len(cases):
        raise ValueError("Covariance dimensions must be unique; select one sweep with a common protocol")
    if len({(case.get("iterations", 10), case.get("transition_radius", 0.95)) for case in cases}) != 1:
        raise ValueError("Covariance sweep must use the same iteration count and transition radius")
    case_ids = {case["id"] for case in cases}
    summary = {}
    for item in summaries:
        if item["case_id"] in case_ids and item["mode"] == "launch":
            key = (item["case_id"], item["backend"])
            if key in summary:
                raise ValueError(f"Duplicate summary entry: {key}")
            if item.get("primary_time_basis") != "steady_wall_ms":
                raise ValueError(f"Summary does not declare application wall time as primary: {key}")
            summary[key] = item
    for case in cases:
        for backend in case.get("backends", config["backends"]):
            if (case["id"], backend) not in summary:
                raise ValueError(f"Missing completed summary for {case['id']}/{backend}")
    return config, manifest, cases, summary, runs, metrics


def metric_extrema(cases, metrics):
    """Maximum over all processes, seeds and saved execution phases."""
    case_ids = {case["id"] for case in cases}
    maxima, thresholds, counts = {}, defaultdict(set), defaultdict(int)
    for row in metrics:
        if row.get("case_id") not in case_ids or row.get("mode") != "launch":
            continue
        values = row.get("values", {})
        for metric in ("relative_error", "symmetry_error"):
            value = values.get(metric)
            if finite(value):
                key = (row["case_id"], row["backend"], metric)
                maxima[key] = max(maxima.get(key, 0), value)
                counts[key] += 1
            bound = values.get("quality_thresholds", {}).get(metric)
            if finite(bound, positive=True):
                thresholds[metric].add(bound)
    for case in cases:
        for metric in ("relative_error", "symmetry_error"):
            bound = case.get("quality", {}).get(metric, 1e-5)
            thresholds[metric].add(bound)
    if any(len(values) != 1 for values in thresholds.values()):
        raise ValueError("Quality thresholds differ across the sweep or recorded validation phases")
    return maxima, {metric: next(iter(values)) for metric, values in thresholds.items()}, counts


def _errorbar(ax, x, summary, prefix, color, marker, label=None):
    point = summary.get(prefix + "speedup_geomean")
    if not finite(point, positive=True):
        return False
    low, high = (summary.get(prefix + "speedup_ci95_" + side) for side in ("low", "high"))
    error = None
    if finite(low, positive=True) and finite(high, positive=True):
        if not low <= point <= high:
            raise ValueError("Archived confidence interval does not contain its point estimate")
        error = [[point - low], [high - point]]
    ax.errorbar([x], [point], yerr=error, color=color, marker=marker, linestyle="none",
                markersize=7, capsize=4, elinewidth=1.6, label=label, zorder=3)
    return True


def make_figure(directory, output_prefix):
    config, manifest, cases, summary, runs, metrics = load_archive(directory)
    maxima, thresholds, validation_counts = metric_extrema(cases, metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), gridspec_kw={"width_ratios": [1.16, 1, 1.1]})
    dimensions = [case["n"] for case in cases]
    positions = list(range(len(cases)))
    for ax in axes:
        ax.set_xticks(positions, [f"{n:,}" for n in dimensions])
        ax.set_xlabel("Square matrix dimension n")
        ax.set_xlim(-0.48, len(cases) - 0.52)
        ax.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    # Discrete scatter points avoid implying measurements between observed sizes.
    latency = axes[0]
    latency.set_title("A  Complete application wall latency", loc="left", fontweight="bold")
    latency.set_yscale("log")
    latency.set_ylabel("Wall time per application (ms, log scale)")
    failure_notes = []
    offsets = [-0.18, -0.06, 0.06, 0.18]
    backend_handles = []
    for offset, (backend, label, color, marker) in zip(offsets, BACKENDS):
        used = False
        for x, case in zip(positions, cases):
            item = summary.get((case["id"], backend), {})
            wall = item.get("median_wall_ms")
            if not finite(wall, positive=True):
                continue
            used = True
            passed = item.get("all_observed_quality_pass") is True
            latency.scatter(x + offset, wall, color=color if passed else "white", edgecolor=color,
                            marker=marker, s=62, linewidth=1.5, zorder=3)
            if not passed:
                latency.scatter(x + offset, wall, color=FAILED_COLOR, marker="x", s=82, linewidth=1.8, zorder=4)
        if used:
            backend_handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7, label=label))
    backend_handles.append(Line2D([], [], color=FAILED_COLOR, marker="x", linestyle="none", markersize=8,
                                  label="Hollow + orange x: quality failed/unchecked"))
    latency.legend(handles=backend_handles, loc="best", fontsize=8, framealpha=0.94)

    speed = axes[1]
    speed.set_title("B  MegaFMM / cuBLASLt comparison", loc="left", fontweight="bold")
    speed.set_ylabel("Paired speedup (cuBLASLt wall / MegaFMM wall)")
    speed.axhline(1, color="#28323C", linestyle="--", linewidth=1, zorder=1)
    timing_label = qualified_label = True
    for x, case in zip(positions, cases):
        item = summary.get((case["id"], "megafmm"), {})
        timing_used = _errorbar(speed, x - 0.09, item, "", TIMING_COLOR, "o",
                              "All execution-valid timing pairs" if timing_label else None)
        qualified_used = _errorbar(speed, x + 0.09, item, "qualified_", QUALIFIED_COLOR, "s",
                                 "Pairs passing candidate + baseline quality" if qualified_label else None)
        timing_label &= not timing_used
        qualified_label &= not qualified_used
        total, qualified = item.get("paired_process_count", 0), item.get("qualified_paired_process_count", 0)
        speed.text(x, -0.13, f"{qualified}/{total} qualified", ha="center", va="top",
                   fontsize=8, transform=speed.get_xaxis_transform())
        if timing_used and not qualified_used:
            speed.scatter(x - 0.09, item["speedup_geomean"], marker="x", s=80,
                          color=FAILED_COLOR, linewidth=1.8, zorder=4)
    speed.legend(loc="best", fontsize=8, framealpha=0.94)
    speed.margins(y=0.24)

    quality = axes[2]
    quality.set_title("C  Maximum saved FP64-check errors", loc="left", fontweight="bold")
    quality.set_yscale("log")
    quality.set_ylabel("Relative error / asymmetry (log scale)")
    max_error = 0
    zero_count = 0
    for backend_index, (backend, label, color, _) in enumerate(BACKENDS[:2]):
        for metric_index, (metric, description, marker) in enumerate((
                ("relative_error", "relative Frobenius error", "o"),
                ("symmetry_error", "symmetry error", "s"))):
            used = False
            for x, case in zip(positions, cases):
                value = maxima.get((case["id"], backend, metric))
                if value == 0:
                    zero_count += 1
                if not finite(value, positive=True):
                    continue
                used = True
                max_error = max(max_error, value)
                xoffset = (backend_index - 0.5) * 0.19 + (metric_index - 0.5) * 0.065
                quality.scatter(x + xoffset, value, color=color, marker=marker, s=45,
                                facecolor=color if metric_index == 0 else "white", linewidth=1.4, zorder=3)
                if value > thresholds[metric]:
                    quality.scatter(x + xoffset, value, color=FAILED_COLOR, marker="x", s=85, linewidth=1.8, zorder=4)
                    if backend == "megafmm":
                        failure_notes.append(f"n={case['n']:,}: MegaFMM max {metric}={value:.4g} > {thresholds[metric]:.4g}")
            if used:
                quality.scatter([], [], color=color, marker=marker, s=45,
                                facecolor=color if metric_index == 0 else "white", linewidth=1.4,
                                label=f"{'cuBLASLt' if backend == 'cublaslt' else 'MegaFMM'} {description}")
    for bound in sorted(set(thresholds.values())):
        quality.axhline(bound, color="#28323C", linestyle="--", linewidth=1.1,
                        label=f"Declared error upper bound: {bound:g}")
    quality.legend(loc="best", fontsize=8, framealpha=0.94)
    if max_error:
        quality.set_ylim(top=max(max_error, *thresholds.values()) * 5)

    runtime = manifest.get("runtime_fingerprint", {}).get("runtime", {})
    driver = manifest.get("runtime_fingerprint", {}).get("driver", "unrecorded")
    measurement = config["measurement"]
    iteration_count = cases[0].get("iterations", 10)
    radius = cases[0].get("transition_radius", 0.95)
    fig.suptitle("FP32 dense covariance propagation: measured performance and numerical quality",
                 x=0.055, y=0.97, ha="left", fontsize=17, fontweight="bold")
    subtitle = (f"P <- (A P) A.T + Q | {iteration_count} steps | transition radius parameter {radius:g} | "
                f"{measurement['independent_processes']} independent processes | launch mode")
    fig.text(0.055, 0.92, subtitle, fontsize=11, color="#48525C")
    caption = [
        ("A: median of independent-process steady wall times; each process uses the median of saved group means. "
         "Markers show only measured dimensions. Hollow orange-cross points did not pass every observed application quality gate."),
        ("B: paired independent-process geometric means and archived 95% bootstrap intervals when available. Gray includes quality failures; "
         "green includes only pairs where both candidate and baseline passed. Counts identify the qualified subset."),
        ("C: maxima over all saved fresh-input and timed-path validation records, with independent FP64 references. "
         "The plotted errors do not replace the separately enforced positive-definiteness gate. TF32 permission does not establish actual TF32 execution."),
    ]
    if failure_notes:
        caption.append("Quality boundary: " + "; ".join(failure_notes) + ".")
    if zero_count:
        caption.append(f"{zero_count} exactly zero error maxima omitted from the logarithmic quality axis.")
    caption.append(f"Device: {runtime.get('gpu', 'unrecorded')}; PyTorch {runtime.get('torch', 'unrecorded')}; "
                   f"CUDA {runtime.get('cuda_runtime', 'unrecorded')}; driver {driver}. "
                   f"Created UTC: {manifest.get('created_utc', 'unrecorded')}; "
                   f"library SHA256: {(manifest.get('library_sha256') or 'unrecorded')[:16]}; source hashes archived in manifest.json.")
    caption.append(f"Archive: {directory}")
    wrapped = "\n".join(textwrap.fill(line, width=186) for line in caption)
    fig.text(0.055, 0.045, wrapped, va="bottom", ha="left", fontsize=8, color="#35404A", linespacing=1.35)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.39, wspace=0.31)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        output = Path(str(output_prefix) + "." + extension)
        fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Completed suite result directory (summary.json must exist)")
    parser.add_argument("--output-prefix", type=Path,
                        help="Output path without extension (default: DIRECTORY/scientific_covariance)")
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    prefix = (args.output_prefix.expanduser().resolve() if args.output_prefix
              else directory / "scientific_covariance")
    try:
        outputs = make_figure(directory, prefix)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
