#!/usr/bin/env python3
"""Report explicitly selected small GEMM measurements without running GPU work.

Example final invocation (list every completed candidate CSV explicitly):
  python3 scripts/summarize_small.py --files results/small_final_lt_launch.csv \
      results/small_final_lt_graph.csv

Original small_baseline_*.csv files are loaded separately. Markdown parsing and
matching use the standard library; matplotlib only exports standalone SVG/PNG.
Existing large-matrix reports, manifests, and plots are never modified.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FLOORS = {"bf16": 1e-3, "fp16": 1e-4}
WORKSPACE_LIMIT = 8 * 1024**3
TARGET = 1.05
DEFAULT_SQUARES = (128, 256, 512, 1024, 2048, 3072, 4096)
REQUIRED = {
    "m", "n", "k", "dtype", "requested", "actual", "variant", "mode",
    "cache", "rotate_inputs", "quantile_unit", "trials", "groups",
    "correct_trials", "workspace_limit", "dense_event_ms", "dense_p10_ms",
    "dense_p90_ms", "dense_cv", "solution_event_ms", "solution_p10_ms",
    "solution_p90_ms", "solution_cv", "dense_ef", "solution_ef", "gate_floor",
    "gate_limit", "nonfinite", "numerical_pass",
}


@dataclass(frozen=True)
class Record:
    source: Path
    line: int
    role: str
    values: dict[str, str]

    def number(self, field: str, default: float = 0.0) -> float:
        value = self.values.get(field, "")
        return float(value) if value else default

    def integer(self, field: str, default: int = 0) -> int:
        value = self.number(field, default)
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"{field} must be a finite integer")
        return int(value)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.integer(key) for key in ("m", "n", "k"))

    @property
    def actual(self) -> str:
        return self.values["actual"]

    @property
    def backend(self) -> str:
        actual = self.actual.lower()
        if actual.startswith(("winograd_", "megakernel_")):
            return "sw"
        if actual.startswith("dense"):
            return "dense" if self.values["requested"] == "dense" else "dense_fallback"
        return "other"

    @property
    def checked(self) -> bool:
        return self.integer("correct_trials") > 0 and self.values["numerical_pass"] in {"0", "1"}

    @property
    def frozen_contract(self) -> bool:
        floor = FLOORS.get(self.values["dtype"])
        return (
            floor is not None
            and math.isclose(self.number("gate_floor"), floor, rel_tol=0, abs_tol=1e-12)
            and math.isfinite(self.number("dense_ef"))
            and math.isclose(self.number("gate_limit"), floor + 3*self.number("dense_ef"),
                             rel_tol=1e-6, abs_tol=1e-12)
        )

    @property
    def numerical_pass(self) -> bool:
        # The flag is the conjunction of every fresh-input trial. Max eF/max gate
        # alone cannot reconstruct that conjunction; require the recorded flag.
        return (
            self.checked and self.values["numerical_pass"] == "1"
            and self.integer("nonfinite") == 0 and self.frozen_contract
            and math.isfinite(self.number("solution_ef"))
            and self.number("solution_ef") <= self.number("gate_limit")
        )

    @property
    def status(self) -> str:
        if not self.checked:
            return "unchecked；不可用于 SW 验收"
        if not self.frozen_contract:
            return "固定门限合同不一致；需检查"
        if self.backend == "dense_fallback":
            return "Dense fallback 通过；非 SW 认证" if self.numerical_pass else "Dense fallback 数值失败"
        if self.backend == "sw":
            return "SW 固定门限通过；仅所测样本" if self.numerical_pass else "SW 数值失败；未认证"
        return "实际后端固定门限通过；非 SW 认证" if self.numerical_pass else "实际后端数值失败"

    @property
    def full_protocol(self) -> bool:
        return (
            self.integer("trials") >= 100 and self.integer("groups") >= 5
            and self.integer("correct_trials") >= 3
            and self.values["quantile_unit"] == "group_mean"
            and self.integer("workspace_limit") == WORKSPACE_LIMIT
        )

    @property
    def speedup(self) -> float:
        return self.number("dense_event_ms") / self.number("solution_event_ms")

    @property
    def dispersion_ratio(self) -> tuple[float, float]:
        return (self.number("dense_p10_ms") / self.number("solution_p90_ms"),
                self.number("dense_p90_ms") / self.number("solution_p10_ms"))

    @property
    def match_key(self) -> tuple:
        # Historical matches require the same precision, shape, execution mode,
        # cache scenario, address rotation, and input working set.
        return (self.values["dtype"], self.shape, self.values["mode"],
                self.values["cache"], self.integer("rotate_inputs"),
                self.integer("input_working_set_bytes", 0))

    @property
    def panel(self) -> tuple:
        category = self.backend if self.checked else f"unchecked_{self.backend}"
        return (self.values["dtype"], self.values["mode"], self.values["cache"],
                self.integer("rotate_inputs"), category)

    @property
    def series(self) -> tuple:
        return self.source, self.actual, self.integer("variant"), self.role

    @property
    def variant_key(self) -> tuple:
        # A source file is an immutable observation set, rather than permission
        # to combine different source revisions with the same variant number.
        return (self.source, self.actual, self.integer("variant"), *self.panel)


def read_records(paths: list[Path], role: str) -> tuple[list[Record], list[str]]:
    records, warnings, seen_paths = [], [], set()
    for path in paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"missing explicitly selected CSV: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
            seen_rows = set()
            for line, values in enumerate(reader, start=2):
                if not values.get("m") or values["m"] == "m":
                    continue
                r = Record(path, line, role, values)
                try:
                    if r.values["dtype"] not in FLOORS or r.values["mode"] not in {"launch", "graph"}:
                        raise ValueError("unknown dtype or execution mode")
                    if min(r.shape) <= 0:
                        raise ValueError("shape dimensions must be positive")
                    _ = r.panel, r.series, r.match_key, r.full_protocol, r.checked
                    for field in ("dense_event_ms", "solution_event_ms", "dense_p10_ms", "dense_p90_ms",
                                  "solution_p10_ms", "solution_p90_ms"):
                        if not math.isfinite(r.number(field)) or r.number(field) <= 0:
                            raise ValueError(f"{field} must be finite and positive")
                    for field in ("dense_cv", "solution_cv"):
                        if not math.isfinite(r.number(field)) or r.number(field) < 0:
                            raise ValueError(f"{field} must be finite and nonnegative")
                    if r.integer("nonfinite") < 0:
                        raise ValueError("nonfinite count cannot be negative")
                except (ValueError, TypeError, OverflowError) as error:
                    raise ValueError(f"{path}:{line}: {error}") from error
                key = r.match_key, r.actual, r.integer("variant")
                if key in seen_rows:
                    warnings.append(f"{path.name}:{line}: duplicate row key; retained as separate evidence")
                seen_rows.add(key)
                if values.get("speedup") and not math.isclose(r.number("speedup"), r.speedup,
                                                               rel_tol=1e-7, abs_tol=1e-10):
                    warnings.append(f"{path.name}:{line}: stored speedup differs; recomputed from same-row times")
                if r.checked and not r.frozen_contract:
                    warnings.append(f"{path.name}:{line}: frozen floor/gate formula mismatch")
                if r.checked and values["numerical_pass"] == "1" and not r.numerical_pass:
                    warnings.append(f"{path.name}:{line}: recorded pass contradicts metrics or frozen contract")
                if r.values["quantile_unit"] != "group_mean":
                    warnings.append(f"{path.name}:{line}: quantile unit differs from the predeclared group-mean protocol")
                records.append(r)
    return records, warnings


class BaselineIndex:
    def __init__(self, records: list[Record]):
        self.route, self.lt0 = defaultdict(list), defaultdict(list)
        for r in records:
            self.route[r.match_key, r.actual].append(r)
            if r.backend == "sw" and r.values["requested"] == "lt" and r.integer("variant") == 0:
                self.lt0[r.match_key].append(r)

    def original_route(self, r: Record) -> Record | None:
        matches = self.route.get((r.match_key, r.actual), ())
        return matches[0] if r.backend == "sw" and len(matches) == 1 else None

    def original_lt(self, r: Record) -> Record | None:
        matches = self.lt0.get(r.match_key, ())
        return matches[0] if r.backend == "sw" and len(matches) == 1 else None

    def warnings(self) -> list[str]:
        result = []
        for key, values in self.route.items():
            if len(values) > 1:
                result.append(f"ambiguous original actual-route baseline {key}: gain reported as N/A")
        for key, values in self.lt0.items():
            if len(values) > 1:
                result.append(f"ambiguous original Lt0 anchor {key}: gain reported as N/A")
        return result


def relative_link(path: Path, target: Path, title: str | None = None) -> str:
    relative = Path(os.path.relpath(path, target.parent)).as_posix()
    return f"[{title or path.name}](<{relative}>)"


def table(cells) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |"


def shape_text(shape: tuple[int, int, int]) -> str:
    return f"{shape[0]}³" if len(set(shape)) == 1 else "×".join(map(str, shape))


def shape_tick(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    return f"{m:,}\n(square)" if m == n == k else f"{m:,}×{n:,}\nK={k:,}"


def shape_order(shape: tuple[int, int, int], squares: tuple[int, ...]) -> tuple:
    if len(set(shape)) == 1 and shape[0] in squares:
        return (0, squares.index(shape[0]), shape)
    return (1, math.prod(shape), shape)


def timing(r: Record, prefix: str) -> str:
    return (f"{1000*r.number(prefix+'_event_ms'):.3f} "
            f"[{1000*r.number(prefix+'_p10_ms'):.3f},{1000*r.number(prefix+'_p90_ms'):.3f}]")


def historic_gain(r: Record, baseline: Record | None) -> str:
    if baseline is None:
        return "N/A"
    low = baseline.number("solution_p10_ms") / r.number("solution_p90_ms")
    high = baseline.number("solution_p90_ms") / r.number("solution_p10_ms")
    return f"{baseline.number('solution_event_ms')/r.number('solution_event_ms'):.3f}× [{low:.3f},{high:.3f}]"


def series_label(series: tuple) -> str:
    source, actual, variant, role = series
    name = source.stem.removeprefix("small_")
    name = name.removesuffix("_launch").removesuffix("_graph")
    name = name.removeprefix("final_")
    short_route = actual.removeprefix("winograd_").removeprefix("megakernel_")
    source = textwrap.fill(f"{'original ' if role == 'baseline' else ''}{name} · v{variant}", width=38)
    return source + "\n" + textwrap.fill(short_route, width=38)


def route_style_key(series: tuple) -> tuple:
    # Source and execution mode remain separate observations. Only their visual
    # encoding is shared so the same route/variant retains its color in all files.
    _, actual, variant, _ = series
    return actual, variant


def select_envelope(records: list[Record]) -> dict[tuple, Record]:
    # This intentionally describes selection after measurement. It is not a
    # fixed dispatcher or evidence of independent out-of-sample performance.
    result = {}
    for r in records:
        if r.backend != "sw" or not r.checked:
            continue
        key = r.panel, r.shape
        previous = result.get(key)
        if previous is None or (r.speedup, str(r.source), -r.line) > (previous.speedup, str(previous.source), -previous.line):
            result[key] = r
    return result


def adjacent_pairs(records: list[Record], squares: tuple[int, ...]) -> list[tuple[Record, Record]]:
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.backend == "sw" and r.role == "candidate" and len(set(r.shape)) == 1 and r.shape[0] in squares:
            groups[r.variant_key][r.shape[0]].append(r)
    pairs = []
    for sizes in groups.values():
        for a, b in zip(squares, squares[1:]):
            # Duplicate observations require explicit curation rather than a
            # best-of-repeat selection that improves acceptance retrospectively.
            if len(sizes.get(a, ())) == len(sizes.get(b, ())) == 1:
                pairs.append((sizes[a][0], sizes[b][0]))
    return pairs


def pair_screen(pair: tuple[Record, Record]) -> tuple[bool, bool, bool]:
    numeric_full = all(r.numerical_pass and r.full_protocol for r in pair)
    median_pass = all(r.speedup >= TARGET for r in pair)
    dispersion_pass = all(r.dispersion_ratio[0] >= TARGET for r in pair)
    return numeric_full, median_pass, dispersion_pass


def export_figures(records: list[Record], candidates: list[Record], baseline: BaselineIndex,
                   directory: Path, squares: tuple[int, ...]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    by_panel = defaultdict(list)
    for r in records:
        by_panel[r.panel].append(r)
    envelope = select_envelope(candidates)
    style_keys = sorted({route_style_key(r.series) for r in records})
    color_map = plt.get_cmap("tab20")
    # Use the saturated tab20 colors first, then the lighter alternatives. There
    # are no source/mode duplicates consuming palette entries.
    palette_order = list(range(0,20,2)) + list(range(1,20,2))
    colors = {key: color_map(palette_order[i % 20]) for i,key in enumerate(style_keys)}
    marks = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    markers = {key: marks[i % len(marks)] for i,key in enumerate(style_keys)}
    artifacts = []
    categories = sorted({p[-1] for p in by_panel})
    primary_shapes = {(n,n,n) for n in (128,256,512,1024)}
    for category in categories:
        panels = sorted((p for p in by_panel if p[-1] == category),
                        key=lambda p: (p[1], p[2], p[3], p[0]))
        scenarios = sorted({(p[1],p[2],p[3]) for p in panels})
        kinds = ("primary_performance", "performance", "numerical", "original_lt_gain") if category == "sw" else ("performance", "numerical")
        for kind in kinds:
            for mode,cache,rotations in scenarios:
                displayed_panels = [p for p in panels if p[1:4] == (mode,cache,rotations)]
                if kind == "primary_performance":
                    displayed_panels = [p for p in displayed_panels if any(r.panel == p and
                                        r.shape in primary_shapes for r in candidates)]
                if not displayed_panels:
                    continue
                groups = []
                for panel in displayed_panels:
                    group = by_panel[panel]
                    if kind == "primary_performance":
                        group = [r for r in group if r.shape in primary_shapes and
                                 (r.role == "candidate" or (r.values["requested"] == "lt" and r.integer("variant") == 0))]
                    groups.append(group)
                # Each mode/cache/address scenario has at most BF16 and FP16.
                # A dedicated second row contains legends; titles and labels
                # cannot collide with the next execution mode's panel.
                cols = len(displayed_panels)
                legend_lines = []
                for group in groups:
                    labels = [series_label(s) for s in sorted({r.series for r in group},
                                                               key=lambda s:tuple(map(str,s)))]
                    labels += ["auxiliary reference"] * (3 if kind in {"performance","primary_performance"} else 2)
                    half = math.ceil(len(labels)/2)
                    legend_lines.append(max(sum(label.count("\n")+1 for label in labels[:half]),
                                            sum(label.count("\n")+1 for label in labels[half:])))
                legend_height = .17*max(legend_lines) + .4
                fig = plt.figure(figsize=(8.2*cols, 4.6+legend_height), constrained_layout=True)
                grid = fig.add_gridspec(2,cols,height_ratios=(4.6,legend_height))
                for column,(panel,group) in enumerate(zip(displayed_panels,groups)):
                    ax = fig.add_subplot(grid[0,column])
                    legend_ax = fig.add_subplot(grid[1,column])
                    legend_ax.set_axis_off()
                    shapes = sorted({r.shape for r in group}, key=lambda s: shape_order(s,squares))
                    positions = {shape:i for i,shape in enumerate(shapes)}
                    series_records = defaultdict(list)
                    for r in group:
                        series_records[r.series].append(r)
                    for series,samples in sorted(series_records.items(),key=lambda item:tuple(map(str,item[0]))):
                        samples.sort(key=lambda r:(positions[r.shape],r.line))
                        if kind == "numerical":
                            samples = [r for r in samples if r.checked and math.isfinite(r.number("solution_ef"))]
                        if kind == "original_lt_gain":
                            samples = [r for r in samples if baseline.original_lt(r) is not None]
                        if not samples:
                            continue
                        xs = [positions[r.shape] for r in samples]
                        if kind in {"performance","primary_performance"}:
                            ys = [r.speedup for r in samples]
                            ranges = [r.dispersion_ratio for r in samples]
                        elif kind == "original_lt_gain":
                            anchors = [baseline.original_lt(r) for r in samples]
                            ys = [a.number("solution_event_ms")/r.number("solution_event_ms") for r,a in zip(samples,anchors)]
                            ranges = [(a.number("solution_p10_ms")/r.number("solution_p90_ms"),
                                       a.number("solution_p90_ms")/r.number("solution_p10_ms")) for r,a in zip(samples,anchors)]
                        else:
                            ys = [max(1e-12,r.number("solution_ef")) for r in samples]
                            ranges = None
                        key = route_style_key(series)
                        style = {"color":colors[key],"marker":markers[key],
                                 "markersize":4,"linewidth":1.25,
                                 "linestyle":"--" if series[-1] == "baseline" else "-",
                                 "label":series_label(series)}
                        if ranges is None:
                            ax.plot(xs,ys,**style)
                        else:
                            ax.errorbar(xs,ys,yerr=([max(0,y-lo) for y,(lo,hi) in zip(ys,ranges)],
                                                   [max(0,hi-y) for y,(lo,hi) in zip(ys,ranges)]),capsize=2.5,**style)
                    if kind in {"performance","primary_performance"} and category == "sw":
                        selected = [envelope[panel,shape] for shape in shapes if (panel,shape) in envelope]
                        if selected:
                            ax.plot([positions[r.shape] for r in selected],[r.speedup for r in selected],
                                    color="black",linestyle="-.",linewidth=1.7,
                                    marker="*",markersize=7,label="Empirical envelope: selected on measurement")
                    dtype,mode,cache,rotations,_ = panel
                    category_text = {"sw":"Actual SW routes; numerical status in tables",
                                     "dense_fallback":"Actual Dense fallback; no SW acceptance",
                                     "dense":"Actual ordinary Dense backend"}.get(category,category)
                    ax.set_title(f"{dtype.upper()} · {mode} · {cache} · pairs={rotations}\n{category_text}",fontsize=11)
                    ax.set_xticks(range(len(shapes)),[shape_tick(shape) for shape in shapes],
                                  fontsize=8,rotation=30,ha="right")
                    ax.set_xlabel("Predeclared shape M × N × K",fontsize=9)
                    ax.grid(axis="y",alpha=.2)
                    if kind in {"performance","primary_performance"}:
                        ax.axhline(1,color="#334155",linestyle="--",linewidth=1,label="Same-row tuned Dense = 1×")
                        if category == "sw":
                            ax.axhline(TARGET,color="#64748b",linestyle=":",linewidth=1,label="1.05× target; accuracy also required")
                        ax.set_ylabel("Same-row Dense / actual backend",fontsize=9)
                        ax.set_ylim(bottom=0)
                    elif kind == "original_lt_gain":
                        ax.axhline(1,color="#334155",linestyle=":",linewidth=1,label="Original Lt0 anchor = 1×")
                        ax.set_ylabel("Original Lt0 / measured SW duration\nHistorical comparison; separate runs",fontsize=9)
                        ax.set_ylim(bottom=0)
                    else:
                        ax.axhline(FLOORS[dtype],color="#111827",linestyle="--",linewidth=1.25,
                                   label=f"Frozen floor = {FLOORS[dtype]:g}")
                        limits = defaultdict(list)
                        for r in group:
                            if r.checked and math.isfinite(r.number("gate_limit")):
                                limits[r.shape].append(r.number("gate_limit"))
                        if limits:
                            ordered = sorted(limits,key=lambda shape:positions[shape])
                            ax.plot([positions[shape] for shape in ordered],[max(limits[shape]) for shape in ordered],
                                    color="#64748b",linestyle=":",linewidth=1,
                                    label="Max reported per-trial gate (display only)")
                        ax.set_yscale("log")
                        ax.set_ylabel("Worst reported relative Frobenius error eF",fontsize=9)
                    handles,labels = ax.get_legend_handles_labels()
                    legend_ax.legend(handles,labels,loc="upper center",ncols=2,
                                     fontsize=7.8,frameon=False,columnspacing=1.2,handlelength=2.2)
                scenario_suffix = mode
                if sum(s[0] == mode for s in scenarios) > 1:
                    scenario_suffix += f"_{cache.replace('/', '_').replace(' ', '_')}_pairs{rotations}"
                for suffix in ("svg","png"):
                    path = directory / f"{kind}_{category}_{scenario_suffix}.{suffix}"
                    fig.savefig(path,dpi=180,bbox_inches="tight")
                    artifacts.append(path)
                plt.close(fig)
    return artifacts


def write_report(records: list[Record], candidates: list[Record], baseline: BaselineIndex,
                 warnings: list[str], artifacts: list[Path], target: Path,
                 suite_path: Path, squares: tuple[int, ...], command: str) -> None:
    candidate_counts = Counter((r.backend, "unchecked" if not r.checked else
                                "pass" if r.numerical_pass else "fail") for r in candidates)
    sw = [r for r in candidates if r.backend == "sw"]
    eligible_rows = [r for r in sw if r.numerical_pass and r.full_protocol and r.speedup >= TARGET]
    pairs = adjacent_pairs(candidates, squares)
    qualifying_pairs = [pair for pair in pairs if all(pair_screen(pair))]
    max_ratio = max((r.speedup for r in sw), default=float("nan"))
    lines = ["# 小矩阵优化实验报告", "",
             f"生成时间：{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}。仅读取明确指定的 CSV；没有执行 GPU 工作。", "",
             f"候选共 {len(candidates)} 行、原始基线 {len(records)-len(candidates)} 行。候选 SW 固定门限通过 {sum(r.numerical_pass for r in sw)}/{len(sw)} 行，"
             f"unchecked {sum(not r.checked for r in candidates)} 行；CSV nonfinite 元素计数合计 {sum(r.integer('nonfinite') for r in candidates)}。"
             f"同时满足固定门限、完整采样配置和单行 ≥1.05× 的 SW 观测为 {len(eligible_rows)} 行；相邻预声明尺寸的保守离散度筛查通过 {len(qualifying_pairs)} 对。", "",
             f"所测 SW 的最大逐行 Dense/候选比为 {max_ratio:.4f}×。数值失败、unchecked 和 Dense fallback 均不参与 SW 验收；未认证路径不能启用 Auto。局部门限或性能筛查通过也不能替代独立结构、sanitizer 和输入范围认证。", "",
             "## 对照合同与复现", "",
             f"预声明尺寸：{relative_link(suite_path,target)}。方阵顺序为 {', '.join(map(str,squares))}，另含两个矩形；输入 BF16/FP16、输出 FP32，行主序、alpha=1、beta=0。", "",
             "完整协议为 `--warmup 20 --trials 100 --groups 5 --correct-trials 3`；快速 `20/5/1` 仅用于诊断。CSV 可核对 trials/groups/correct_trials、8 GiB workspace 上限和 group_mean；warmup 未写入 CSV，须从原始命令/日志核对。Dense 独立调优，所有输入相关变换、叶乘法和重组均在完整调用计时内，Graph 捕获同样包含这些阶段。调优、输入生成和 FP64 参考位于计时区间外。", "",
             "launch 与 Graph、BF16 与 FP16、warm_cache 与 rotating_inputs 分开报告。median、p10/p90 和 CV 基于五组每调用平均时间，不是单次调用分布。性能区间 `[Dense p10 / 候选 p90, Dense p90 / 候选 p10]` 表示离散度，不能解释为置信区间。", "",
             "固定数值合同为每次新的目标精度输入直接转换到 FP64，计算原始 `AB` 参考；逐试验要求 `eF_solution ≤ 3·eF_Dense + floor`，BF16 floor=1e-3，FP16 floor=1e-4，并要求双方无 nonfinite。CSV eF 和 gate_limit 各自保存最大值，数值是否通过以全部试验的 conjunction 标志为准；只比较两个最大值不足以重新证明逐试验通过。", "",
             "原始实际路径改善只匹配 actual route、M/N/K、dtype、mode、cache、输入地址对数和工作集；variant 不要求相同，这是同一实际 route 家族的新旧 variant 历史比较，双方 variant/source 在逐行表中给出。原始 Lt0 改善单独列为跨路径历史锚点。多个匹配来源时不选择最快基线，而报告 N/A。历史改善使用不同运行的绝对耗时，受时钟、调优算法、共享干扰和 host submission 差异影响；同一行 Dense 对照是性能验收的主要比较。", "",
             "```text", command, "```", "",
             "## 来源与数值分类", "",
             table(["角色", "CSV / SHA256", "行数", "配置 trials/groups/correct", "实际 route / variant"]),
             table(["---", "---", "---:", "---", "---"])]
    by_source = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    for path, group in sorted(by_source.items()):
        configs = sorted({f"{r.integer('trials')}/{r.integer('groups')}/{r.integer('correct_trials')}" for r in group})
        routes = sorted({f"{r.actual} v{r.integer('variant')}" for r in group})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(table([group[0].role, f"{relative_link(path,target)}<br>`{digest}`", len(group),
                            "; ".join(configs), "; ".join(routes)]))
    lines += ["", table(["角色", "实际后端", "数值分类", "行数"]), table(["---", "---", "---", "---:"])]
    classification_counts = Counter((r.role,r.backend,"unchecked" if not r.checked else
                                   "pass" if r.numerical_pass else "fail") for r in records)
    for (role,backend,status), count in sorted(classification_counts.items()):
        lines.append(table([role,backend,status,count]))
    lines += ["", "通过的 fallback 行只说明普通 GEMM 路由通过所测数值样本，不能算作 SW 认证或 SW speed acceptance。显式候选 CSV 的文件名不证明源码身份；对应日志、命令、源码快照/commit 和 telemetry 应与最终归档一起核对。", ""]
    for dtype in ("bf16", "fp16"):
        checked = [r for r in candidates if r.values["dtype"] == dtype and r.checked]
        finite_sw = [r.number("solution_ef") for r in checked if r.backend == "sw" and math.isfinite(r.number("solution_ef"))]
        if checked:
            errors = f"{min(finite_sw):.6g}–{max(finite_sw):.6g}" if finite_sw else "无有限 SW 数据"
            lines += [f"{dtype.upper()}：候选已检查 {len(checked)} 行，实际 SW eF 范围 {errors}；floor={FLOORS[dtype]:g}。", ""]
    lines += ["## 128–1024 主表：测量后选择的经验包络", "",
              "每个 dtype/mode/cache/地址对数/形状，在明确指定候选中的已检查 SW 行选择最大逐行 Dense/候选比；此包络为 **selected-on-measurement**，可能逐尺寸切换 variant，也会受到 Dense 调优/运行波动影响。它不是预先固定的 dispatcher、独立验证收益或数值认证。主表耗时和比值仅列组均值分布的 median；完整逐行表保留所有原始实际 route、来源、p10/p90、CV 和数值结果，包括失败和 unchecked。", ""]
    envelope = select_envelope(candidates)
    panels = sorted({r.panel for r in sw if r.checked}, key=lambda p:(p[1],p[2],p[3],p[0]))
    for panel in panels:
        dtype, mode, cache, rotations, _ = panel
        lines += [f"### {dtype.upper()} / {mode} / {cache} / pairs={rotations}", "",
                  table(["M,N,K", "原始 Lt0 µs", "候选 / source:row", "同一行 Dense µs", "候选 µs", "Dense/候选", "原始 Lt0/候选", "数值状态"]),
                  table(["---", "---:", "---", "---:", "---:", "---:", "---:", "---"])]
        for size in (128,256,512,1024):
            shape = (size,size,size)
            r = envelope.get((panel,shape))
            if r is None:
                lines.append(table([shape_text(shape), "—", "缺少明确候选", "—", "—", "—", "—", "未检查"])); continue
            original = baseline.original_lt(r)
            source_label = r.source.name.removeprefix("small_final_").removeprefix("small_")
            source_label = "<br>".join(textwrap.wrap(source_label,width=26))
            candidate = f"{r.values['requested']} v{r.integer('variant')}<br>{relative_link(r.source,target,source_label)}:{r.line}"
            original_time = f"{1000*original.number('solution_event_ms'):.3f}" if original else "N/A"
            gain = f"{original.number('solution_event_ms')/r.number('solution_event_ms'):.3f}×" if original else "N/A"
            lines.append(table([shape_text(shape),original_time,candidate,
                                f"{1000*r.number('dense_event_ms'):.3f}",f"{1000*r.number('solution_event_ms'):.3f}",
                                f"{r.speedup:.3f}×",gain,r.status]))
        lines.append("")
    lines += ["## 全尺寸、全实际路径证据表", "",
              "耗时单位均为 µs，格式为 median [p10,p90]；历史改善格式为 ratio [p10/p90 端点]。每组先按实际 route 再按预声明尺寸排序；Dense fallback 与 SW 分开。", ""]
    by_panel = defaultdict(list)
    for r in records:
        by_panel[r.panel].append(r)
    for panel, group in sorted(by_panel.items(), key=lambda item:(item[0][4],item[0][1],item[0][2],item[0][3],item[0][0])):
        dtype,mode,cache,rotations,category = panel
        lines += [f"### {dtype.upper()} / {mode} / {cache} / pairs={rotations} / {category}", "",
                  table(["角色 / source:row / 实际 route", "M,N,K", "Dense µs", "实际后端 µs", "Dense/实际 [离散度]", "原始同 route/当前", "原始 Lt0/当前", "CV Dense/实际", "eF Dense/实际", "floor / gate", "数值状态 / 完整采样"]),
                  table(["---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "---"])]
        for r in sorted(group, key=lambda r:(r.actual,r.integer("variant"),shape_order(r.shape,squares),r.role,str(r.source),r.line)):
            low,high = r.dispersion_ratio
            lines.append(table([f"{r.role} / {relative_link(r.source,target)}:{r.line} / {r.actual} v{r.integer('variant')}",
                                shape_text(r.shape), timing(r,"dense"), timing(r,"solution"),
                                f"{r.speedup:.4f}× [{low:.4f},{high:.4f}]",
                                historic_gain(r,baseline.original_route(r)), historic_gain(r,baseline.original_lt(r)),
                                f"{100*r.number('dense_cv'):.2f}%/{100*r.number('solution_cv'):.2f}%",
                                f"{r.number('dense_ef'):.3e}/{r.number('solution_ef'):.3e}",
                                f"{r.number('gate_floor'):.2e}/{r.number('gate_limit'):.3e}",
                                f"{r.status} / {'full CSV protocol' if r.full_protocol else 'diagnostic/incomplete'}"]))
        lines.append("")
    lines += ["## 相邻预声明尺寸的保守验收筛查", "",
              f"预声明相邻方阵对为 {'; '.join(f'{a}–{b}' for a,b in zip(squares,squares[1:]))}。在同一 CSV/实际 route/variant/dtype/mode/cache/地址对数下，两个尺寸均须通过固定逐试验数值门限和完整采样配置，median ratio ≥1.05。额外报告 `Dense p10 / 候选 p90 ≥1.05` 的保守离散度筛查；它不是统计置信界，也不是额外数值门限。", "",
              f"可匹配相邻对 {len(pairs)}；两个 median 均达到目标 {sum(pair_screen(p)[1] for p in pairs)}；同时通过数值与完整采样 {sum(pair_screen(p)[0] and pair_screen(p)[1] for p in pairs)}；再通过保守离散度筛查 {len(qualifying_pairs)}。独立重复、结构/内存检查、适用范围和源码身份仍需核对，脚本不修改 Auto。", ""]
    median_pairs = [pair for pair in pairs if pair_screen(pair)[1]]
    if median_pairs:
        lines += [table(["source / actual / variant", "dtype / mode", "相邻尺寸", "median ratios", "p10/p90 ratios", "固定数值+full", "保守筛查"]),
                  table(["---", "---", "---", "---", "---", "---", "---"])]
        for pair in median_pairs:
            a,b=pair; numeric_full,median_pass,dispersion_pass=pair_screen(pair)
            lines.append(table([f"{relative_link(a.source,target)} / {a.actual} v{a.integer('variant')}",
                                f"{a.values['dtype']} / {a.values['mode']}",f"{a.shape[0]}–{b.shape[0]}",
                                f"{a.speedup:.3f}×/{b.speedup:.3f}×",f"{a.dispersion_ratio[0]:.3f}×/{b.dispersion_ratio[0]:.3f}×",
                                "通过" if numeric_full else "未通过", "通过局部筛查" if all(pair_screen(pair)) else "未通过"]))
        lines.append("")
    lines += ["## 独立静态图与原始证据", ""]
    lines += [relative_link(path,target) for path in artifacts]
    lines += ["", "每种 mode/cache/输入地址配置分别输出一张 BF16/FP16 并排图，图例位于专用行；同一 actual route/variant 在各模式保留颜色和 marker，来源和观测仍分别列出，不合并统计。主性能图仅展示 128–1024 的明确候选与原始 Lt0；全九尺寸图保留全部原始路径。每条曲线明确标注实际 route、variant 和来源；baseline 为虚线。SW、普通 Dense fallback、unchecked 分面。误差条表示上述组均值离散度，不是置信区间。原始 Lt0 改善图是跨运行历史比较；数值图精确零仅为 log 显示置于 1e-12，灰线最大 gate 仅用于显示，不能取代逐行/逐试验认证。", ""]
    for path in sorted(by_source):
        evidence = [sibling for sibling in (path.with_suffix(".log"), path.with_suffix(".exit")) if sibling.is_file()]
        if evidence:
            lines += [f"{relative_link(path,target)} 的运行证据："+", ".join(relative_link(item,target) for item in evidence)+"。", ""]
    if warnings:
        lines += ["## 数据检查提示", ""] + [f"- {warning}" for warning in warnings] + [""]
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", type=Path, nargs="+", required=True,
                        help="Explicit completed candidate CSVs; final report uses results/small_final_*.csv")
    parser.add_argument("--baseline-files", type=Path, nargs="+",
                        help="Original CSVs, default results/small_baseline_*.csv; never inferred from candidate files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--suite", type=Path, default=Path("bench/small_shapes.json"))
    parser.add_argument("--output", type=Path, default=Path("results/SMALL_REPORT.md"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/small_plots"))
    parser.add_argument("--no-plots", action="store_true", help="Write Markdown using only the standard library")
    args = parser.parse_args()
    try:
        baseline_paths = args.baseline_files or sorted(args.results_dir.glob("small_baseline_*.csv"))
        if not baseline_paths:
            raise ValueError("no original small_baseline_* files; supply --baseline-files")
        baseline_records, baseline_warnings = read_records(baseline_paths,"baseline")
        candidates, candidate_warnings = read_records(args.files,"candidate")
        if not candidates:
            raise ValueError("no candidate rows")
        overlap = {r.source for r in baseline_records} & {r.source for r in candidates}
        if overlap:
            raise ValueError(f"candidate and original baseline sources overlap: {', '.join(map(str,overlap))}")
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        squares = tuple(shape[0] for shape in suite.get("squares",())) or DEFAULT_SQUARES
        if len(set(squares)) != len(squares) or any(n <= 0 for n in squares):
            raise ValueError("invalid predeclared square sequence")
    except (ValueError,OSError) as error:
        parser.error(str(error))
    baseline = BaselineIndex(baseline_records)
    records = baseline_records + candidates
    warnings = baseline_warnings + candidate_warnings + baseline.warnings()
    for r in candidates:
        if r.backend == "sw" and baseline.original_lt(r) is None:
            warnings.append(f"{r.source.name}:{r.line}: original Lt0 anchor missing/ambiguous; gain N/A")
    artifacts = [] if args.no_plots else export_figures(records,candidates,baseline,args.plots_dir.resolve(),squares)
    import shlex
    command = " ".join(shlex.quote(arg) for arg in ["python3","scripts/summarize_small.py",*__import__("sys").argv[1:]])
    write_report(records,candidates,baseline,warnings,artifacts,args.output.resolve(),args.suite.resolve(),squares,command)
    print(f"Wrote {args.output}: {len(candidates)} candidate rows, {len(baseline_records)} original rows")
    for artifact in artifacts:
        print(f"Wrote {artifact}")
    for warning in warnings:
        print(f"WARNING {warning}")


if __name__ == "__main__":
    main()
