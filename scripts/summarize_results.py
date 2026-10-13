#!/usr/bin/env python3
"""Export evidence tables and static figures without pooling incompatible GEMMs.

CSV parsing, matching, and report generation use only the Python standard library.
Matplotlib is used solely to export standalone SVG and PNG figures. Every speedup
comes from the tuned dense baseline in the same CSV row. No GPU workload is run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REQUIRED = {"m", "n", "k", "dtype", "actual", "variant", "mode", "cache",
            "trials", "groups", "correct_trials", "dense_event_ms",
            "solution_event_ms", "dense_cv", "solution_cv", "dense_ef",
            "solution_ef", "gate_floor", "gate_limit", "nonfinite", "numerical_pass"}


@dataclass
class Record:
    source: Path
    label: str
    line: int
    values: dict[str, str]

    def number(self, field: str, default: float = 0.0) -> float:
        text = self.values.get(field, "")
        return float(text) if text else default

    def integer(self, field: str, default: int = 0) -> int:
        return int(self.number(field, default))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.integer("m"), self.integer("n"), self.integer("k")

    @property
    def backend(self) -> str:
        actual = self.values["actual"].lower()
        if actual.startswith(("winograd_", "megakernel_")):
            return "sw"
        if actual.startswith("dense"):
            return "dense" if self.values.get("requested") == "dense" else "dense_fallback"
        return "other"

    @property
    def panel(self) -> tuple[str, str, str, int, str]:
        category = self.backend if self.checked else f"unchecked:{self.backend}"
        return (self.values["dtype"], self.values["mode"], self.values["cache"],
                self.integer("rotate_inputs", 1), category)

    @property
    def series(self) -> tuple[str, str, int]:
        return self.label, self.values["actual"], self.integer("variant")

    @property
    def checked(self) -> bool:
        return self.integer("correct_trials") > 0 and self.values["numerical_pass"] in ("0", "1")

    @property
    def passed(self) -> bool:
        return (self.checked and self.values["numerical_pass"] == "1" and self.integer("nonfinite") == 0
                and math.isfinite(self.number("solution_ef"))
                and self.number("solution_ef") <= self.number("gate_limit"))

    @property
    def frozen_gate(self) -> bool:
        floor = {"bf16": 1e-3, "fp16": 1e-4}.get(self.values["dtype"])
        return floor is not None and math.isclose(self.number("gate_floor"), floor, rel_tol=0, abs_tol=1e-12)

    @property
    def certified_sw(self) -> bool:
        return self.backend == "sw" and self.passed and self.frozen_gate

    @property
    def sw_performance_candidate(self) -> bool:
        # This marks an eligible individual observation, not neighboring-shape acceptance.
        return self.certified_sw and self.full_config and self.speedup >= 1.05

    @property
    def verdict(self) -> str:
        if not self.checked:
            return "unchecked；不参与 SW 验收"
        if self.backend == "sw":
            if self.certified_sw:
                return "SW 固定数值门槛通过"
            return "SW 数值失败" if not self.passed else "SW floor 合同不一致"
        if self.backend == "dense_fallback":
            return "Dense fallback 通过；非 SW 认证" if self.passed else "Dense fallback 失败"
        return "普通/其他后端通过；非 SW 认证" if self.passed else "普通/其他后端失败"

    @property
    def full_config(self) -> bool:
        return self.integer("trials") >= 100 and self.integer("groups") >= 5 and self.integer("correct_trials") >= 3

    @property
    def speedup(self) -> float:
        return self.number("dense_event_ms") / self.number("solution_event_ms")


def source_label(path: Path) -> str:
    if path.parent.name.startswith("20") and path.name == "results.csv":
        parts = path.parent.name.split("_", 2)
        return f"{parts[-1]}@{parts[1]}" if len(parts) == 3 else path.parent.name
    return path.stem


def read_records(paths: list[Path]) -> tuple[list[Record], list[str]]:
    records, warnings = [], []
    content_sources = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in content_sources:
            warnings.append(f"{path}: byte-identical CSV mirror of {content_sources[digest]}; excluded from counts and plots")
            continue
        content_sources[digest] = path
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
                warnings.append(f"{path}: missing required columns; excluded")
                continue
            seen = set()
            for line, values in enumerate(reader, start=2):
                if not values.get("m") or values["m"] == "m":
                    continue
                r = Record(path, source_label(path), line, values)
                try:
                    _ = r.shape, r.panel, r.series, r.speedup
                    if r.number("dense_event_ms") <= 0 or r.number("solution_event_ms") <= 0:
                        raise ValueError("timings must be positive")
                    if any(not math.isfinite(r.number(k)) for k in
                           ("dense_event_ms", "solution_event_ms", "dense_cv", "solution_cv")):
                        raise ValueError("timings/CV must be finite")
                except (ValueError, ZeroDivisionError, TypeError) as error:
                    warnings.append(f"{path}:{line}: {error}; excluded")
                    continue
                key = r.shape, r.panel, r.series
                if key in seen:
                    warnings.append(f"{path}:{line}: repeated matching key; plotted independently by row")
                seen.add(key)
                csv_speedup = r.number("speedup", r.speedup)
                if not math.isclose(csv_speedup, r.speedup, rel_tol=1e-7, abs_tol=1e-10):
                    warnings.append(f"{path}:{line}: stored speedup disagrees with same-row times; recomputed")
                if r.checked:
                    frozen = {"bf16": 1e-3, "fp16": 1e-4}.get(r.values["dtype"])
                    if frozen is not None and not math.isclose(r.number("gate_floor"), frozen, rel_tol=0, abs_tol=1e-12):
                        warnings.append(f"{path}:{line}: gate floor differs from fixed contract")
                    if r.values["numerical_pass"] == "1" and r.number("solution_ef") > r.number("gate_limit"):
                        warnings.append(f"{path}:{line}: recorded pass contradicts error/limit; manual review required")
                records.append(r)
    return records, warnings


def discover(results: Path, trajectories: Path) -> list[Path]:
    candidates = [results / "baseline.csv"]
    candidates += sorted(trajectories.glob("*/results.csv"))
    candidates += sorted(results.glob("final*.csv"))
    # current.csv is intentionally excluded: it is a mutable duplicate of a snapshot.
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def shape_label(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    return f"{m}³" if m == n == k else f"{m}×{n}×{k}"


def relative_link(path: Path, report: Path, title: str | None = None) -> str:
    relative = Path(os.path.relpath(path, report.parent)).as_posix()
    return f"[{title or path.name}](<{relative}>)"


def export_figures(records: list[Record], directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    by_panel = defaultdict(list)
    for r in records:
        by_panel[r.panel].append(r)
    panels = sorted(by_panel, key=lambda p: (p[4], p[1], p[2], p[3], p[0]))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    output = []
    for kind in ("performance", "numerical"):
        cols = 2 if len(panels) > 1 else 1
        rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 4.7 * rows),
                                 squeeze=False, constrained_layout=True)
        for index, panel in enumerate(panels):
            ax = axes.flat[index]
            group = by_panel[panel]
            shapes = sorted({r.shape for r in group}, key=lambda s: (math.prod(s), s))
            positions = {shape: i for i, shape in enumerate(shapes)}
            by_series = defaultdict(list)
            for r in group:
                by_series[r.series].append(r)
            for series, samples in sorted(by_series.items()):
                samples = sorted(samples, key=lambda r: (positions[r.shape], r.line))
                if kind == "numerical":
                    samples = [r for r in samples if r.checked]
                if not samples:
                    continue
                xs = [positions[r.shape] for r in samples]
                label, algorithm, variant = series
                legend = f"{label}: {algorithm}, v{variant}"
                if kind == "performance":
                    ys = [r.speedup for r in samples]
                    low, high = [], []
                    for r, y in zip(samples, ys):
                        lower = r.number("dense_p10_ms", r.number("dense_event_ms")) / r.number("solution_p90_ms", r.number("solution_event_ms"))
                        upper = r.number("dense_p90_ms", r.number("dense_event_ms")) / r.number("solution_p10_ms", r.number("solution_event_ms"))
                        low.append(max(0, y-lower)); high.append(max(0, upper-y))
                    ax.errorbar(xs, ys, yerr=(low,high), marker="o", markersize=4,
                                linewidth=1.4, capsize=3, label=legend)
                else:
                    # Exact-zero cases appear at a documented display floor on the logarithmic axis.
                    display_floor = 1e-12
                    ys = [max(display_floor, r.number("solution_ef")) for r in samples]
                    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.4, label=legend)
            dtype, mode, cache, rotations, category = panel
            ax.set_title(f"{dtype.upper()} · {mode} · {cache} · pairs={rotations}\nBackend/evidence: {category}")
            ax.set_xticks(range(len(shapes)), [shape_label(s) for s in shapes], rotation=20, ha="right")
            ax.set_xlabel("Shape M × N × K (each point has its own tuned dense baseline)")
            ax.grid(axis="y", alpha=.22)
            if kind == "performance":
                ax.axhline(1.0, color="#111827", linewidth=1.2, linestyle="--", label="Dense baseline = 1×")
                if category == "sw":
                    ax.axhline(1.05, color="#64748b", linewidth=.9, linestyle=":", label="SW candidate target 1.05×; accuracy also required")
                ax.set_ylabel("Same-row dense / executed-backend event median")
                ax.set_ylim(bottom=0)
            else:
                floor = {"bf16": 1e-3, "fp16": 1e-4}.get(dtype)
                if floor is not None:
                    ax.axhline(floor, color="#111827", linewidth=1.2, linestyle="--", label=f"Frozen floor = {floor:g}")
                # There may be several dense errors per shape; preserve each source's gate separately.
                for series, samples in sorted(by_series.items()):
                    checked = [r for r in samples if r.checked]
                    if checked:
                        checked.sort(key=lambda r: positions[r.shape])
                        ax.plot([positions[r.shape] for r in checked],
                                [max(1e-12,r.number("gate_limit")) for r in checked],
                                color="#94a3b8", alpha=.45, linestyle=":", linewidth=1)
                ax.set_yscale("log")
                ax.set_ylabel("Worst trial Frobenius relative error eF")
                if category.startswith("unchecked:"):
                    ax.text(.5,.5,"Correctness not checked; no SW certification evidence",
                            transform=ax.transAxes,ha="center",va="center",fontsize=9)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, loc="best", framealpha=.85)
        for index in range(len(panels), rows*cols):
            axes.flat[index].set_visible(False)
        title = ("SW experiments and dense fallback shown separately; unchecked rows are ineligible" if kind == "performance"
                 else "Original rounded inputs → FP64 plain GEMM reference; fixed precision gates")
        fig.suptitle(title, fontsize=14)
        for extension in ("svg", "png"):
            path = directory / f"{kind}.{extension}"
            fig.savefig(path, dpi=180)
            output.append(path)
        plt.close(fig)
    return output


def table_line(cells: list[object]) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def stage_profile(path: Path) -> tuple[dict[str, tuple[float,int,int]], str | None]:
    if not path.exists():
        return {}, None
    stats = defaultdict(lambda: [0.0,0,0])
    shape = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("PROFILE "):
            shape = line[len("PROFILE "):]
        parts = line.split()
        if len(parts) < 9:
            continue
        category = "combine" if "combine_kernel" in line else "pack" if "pack_kernel" in line else "leaves" if "sm90_xmma_gemm" in line else None
        if category is None:
            continue
        try:
            percentage, nanoseconds, instances = float(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        stats[category][0] += percentage
        stats[category][1] += nanoseconds
        stats[category][2] += instances
    return dict(stats), shape


def report(records: list[Record], warnings: list[str], artifacts: list[Path],
           target: Path, results: Path) -> None:
    lines = ["# MegaFMM 实验与验收记录", "",
             f"本文件由 `scripts/summarize_results.py` 生成，时间 {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}。", "",
             "**未获得数值认证的强制 SW 路径只能作为研究候选，不能进入 Auto。Auto 保持调优普通 GEMM；性能结果不能替代数值验收。未经正确性检查的 `unchecked` 行也不能作为启用快速路径的依据。**", "",
             "每个 speedup 都由同一 CSV 行的调优普通 GEMM 与实际执行后端的完整调用计算，精度、形状、launch/Graph 模式、warm-cache/轮换输入模式相同。不同 dtype、执行模式和 SW/dense fallback 后端分别展示，未跨模式汇总或计算总体平均加速比。", ""]
    final = [r for r in records if r.source.name.startswith("final")]
    if final:
        lines += [f"已发现最终命名数据集 {len({r.source for r in final})} 个、{len(final)} 行；这些命名表示复测来源，不自动表示性能或精度验收通过。", ""]
    else:
        lines += ["尚无最终命名数据集；当前表格包含初始基线和探索性轨迹。16384 方阵短测中的约 1.04–1.05× 结果是待完整复测的候选观察，时钟和重复组离散度可能影响结论，不能据此宣称已经确认最佳实现。", ""]
    checked = [r for r in records if r.checked]
    sw_checked = [r for r in checked if r.backend == "sw"]
    fallback_checked = [r for r in checked if r.backend == "dense_fallback"]
    other_checked = [r for r in checked if r.backend not in ("sw", "dense_fallback")]
    lines += [f"SW 已检查 {len(sw_checked)} 行：{sum(r.certified_sw for r in sw_checked)} 行固定数值门槛通过，{sum(not r.certified_sw for r in sw_checked)} 行未通过。Dense fallback 已检查 {len(fallback_checked)} 行：{sum(r.passed for r in fallback_checked)} 行 fallback 通过，{sum(not r.passed for r in fallback_checked)} 行失败；这些通过均不是 SW 认证。普通/其他后端已检查 {len(other_checked)} 行；另有 {sum(not r.checked for r in records)} 行 unchecked，不参与 SW 验收。", "",
              f"满足固定 SW 数值门槛、完整 `100/5/3` 或更充分计时配置且单行 speedup ≥ 1.05 的候选观察为 {sum(r.sw_performance_candidate for r in records)} 行。单行候选还需要预先指定的相邻大尺寸和超出噪声的重复证据，不能直接视为性能验收通过。", "",
              "原始输入先舍入至 BF16/FP16，再直接转 FP64 计算普通 GEMM 参考，参考输入不经过 S/T 变换。`actual` 字段决定后端分类；请求 SW 后实际执行 `dense_cublaslt` 的边界、未知范围或资源 fallback 结果只验证普通 GEMM 路由。", "",
              "固定门槛为每次独立试验 `eF_SW ≤ 3*eF_dense + floor`，BF16 floor 为 `1e-3`，FP16 为 `1e-4`。门槛未为某个候选实现放宽；CSV 的误差和 gate_limit 是跨正确性试验的最大值，`numerical_pass` 是逐试验验收的合取。最大误差试验与最大门槛试验可能不同，不能仅用两个最大值替代逐试验验收。", "",
              "## 原始数据与计时配置", "",
              table_line(["来源", "行数", "配置 trials/groups/correct", "模式", "结果状态"]),
              table_line(["---", "---:", "---", "---", "---"])]
    by_source = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    for source, group in by_source.items():
        configs = sorted({f"{r.integer('trials')}/{r.integer('groups')}/{r.integer('correct_trials')}" for r in group})
        modes = sorted({f"{r.values['mode']}/{r.values['cache']}/pairs={r.integer('rotate_inputs',1)}" for r in group})
        state = "完整计时配置" if all(r.full_config for r in group) else "探索性配置，需复测"
        if any(r.checked and r.backend == "sw" and not r.certified_sw for r in group):
            state += "；SW 数值失败"
        if any(r.checked and r.backend == "dense_fallback" for r in group):
            state += f"；fallback 通过 {sum(r.backend=='dense_fallback' and r.passed for r in group)} 行（非 SW 认证）"
        if any(r.checked and r.backend != "sw" and not r.passed for r in group):
            state += "；普通/fallback 数值失败"
        if any(not r.checked for r in group):
            state += "；包含未检查行"
        lines.append(table_line([relative_link(source,target,source_label(source)), len(group),
                                 "; ".join(configs), "; ".join(modes), state]))
    lines += ["", "`current.csv` 是可变文件，与轨迹快照重复，因此自动发现时不纳入。默认自动发现包括全部实验轨迹，不代表最终候选选择；发布最终验收表时使用 `--files` 明确列出已经完成的最终 CSV。轨迹目录保存各自源码快照和输出；跨来源时钟/共享干扰可能不同，同形状跨来源的绝对时间不能作为受控配对实验。", "",
              "## 完整调用的逐行性能与精度", ""]
    by_panel = defaultdict(list)
    for r in records:
        by_panel[r.panel].append(r)
    for panel, group in sorted(by_panel.items(),key=lambda item:(item[0][4],item[0][1],item[0][2],item[0][3],item[0][0])):
        dtype,mode,cache,rotations,category = panel
        solution_label="SW" if category=="sw" else "实际后端"
        lines += [f"### {dtype.upper()} / {mode} / {cache} / input pairs={rotations} / {category}", "",
                  table_line(["来源 / 实际算法", "M,N,K", "Dense median [p10,p90] ms", f"{solution_label} median [p10,p90] ms", "同一行时间比", f"CV dense/{solution_label}", f"eF dense/{solution_label}", "fixed floor / max gate", "分类与数值状态", f"workspace dense/{solution_label} GiB"]),
                  table_line(["---", "---", "---", "---", "---:", "---", "---", "---", "---", "---"])]
        for r in sorted(group,key=lambda x:(math.prod(x.shape),x.shape,x.label,x.line)):
            v=r.values
            timing=lambda prefix:f"{r.number(prefix+'_event_ms'):.4f} [{r.number(prefix+'_p10_ms'):.4f},{r.number(prefix+'_p90_ms'):.4f}]"
            lines.append(table_line([f"{r.label} / {v['actual']} v{r.integer('variant')}",
                                     ",".join(map(str,r.shape)),timing("dense"),timing("solution"),f"{r.speedup:.4f}×",
                                     f"{100*r.number('dense_cv'):.2f}% / {100*r.number('solution_cv'):.2f}%",
                                     f"{r.number('dense_ef'):.3e} / {r.number('solution_ef'):.3e}",
                                     f"{r.number('gate_floor'):.2e} / {r.number('gate_limit'):.3e}",r.verdict,
                                     f"{r.number('dense_workspace')/2**30:.3f} / {r.number('solution_workspace')/2**30:.3f}"]))
        lines.append("")
    lines += ["## 静态导出图", ""]
    for artifact in artifacts:
        lines.append(relative_link(artifact,target))
    lines += ["", "性能图使用逐行加速比，误差条端点为 `dense_p10 / solution_p90` 与 `dense_p90 / solution_p10`，用于表示组计时离散度，不能解释为置信区间。数值图实线为每来源最差正确性试验 eF，黑虚线为固定 floor，灰点线为各来源最大试验 gate_limit；精确零误差仅为对数显示而绘制于 `1e-12`。", "",
              "## 计时和误差解释", "",
              "计时采用预分配、预调优的完整调用；实际执行 SW 的行每次重新执行依赖 A/B 的变换、全部七个叶乘法和 FP32 重组，fallback 行完整执行普通 GEMM。库调优、FP64 参考、输入更新在计时区间之外。Graph 捕获包含实际后端的完整调用；Graph 与普通 launch 的结果分别报告。beta 非零时 C0 语义在功能测试中独立覆盖。", "",
              "CSV 的 median、p10、p90 和 CV 来自独立重复组的每调用平均时间（`quantile_unit=group_mean`），而非单次 invocation 的延迟分布。speedup 是两组 median 的比值。只有五组时尾分位数和 CV 的估计有限；接近 1.05× 的收益必须结合配对重复结果、时钟和噪声验证。", "",
              "warm_cache 重复使用同一对地址，仍每次重算变换。rotating_inputs 轮换多个不同的 A/B 地址对，记录实际输入工作集；应根据工作集与 L2 的大小判断缓存影响，不能仅凭轮换标签断言完全冷缓存。轮换数据不与 warm-cache 合并。", "",
              "原始 CSV 还保留 maxabs、Frobenius eF、`|error|/(atol+rtol*|reference|)` 最大值、`||A||F||B||F` 归一化误差、近零参考的绝对误差、nonfinite 数量、主机 wall time、实际 workspace 与初始化/调优成本。等效 TFLOP/s 使用 `2MNK/T`，不能等同七叶的实际 Tensor Core 算术吞吐。", "",
              "## 工具与独立证据", ""]
    memcheck=results/"memcheck_iter1.txt"
    correctness=results/"correctness_initial.txt"
    if correctness.exists():
        text=correctness.read_text(encoding="utf-8",errors="replace")
        verdict="结构和接口测试通过" if "PASS structural DAG" in text else "需检查日志状态"
        lines += [f"{verdict}：{relative_link(correctness,target)}。结构正确性与混合精度数值认证是不同关口。", ""]
    if memcheck.exists():
        text=memcheck.read_text(encoding="utf-8",errors="replace")
        verdict="compute-sanitizer memcheck 报告 0 errors" if "ERROR SUMMARY: 0 errors" in text else "compute-sanitizer 日志需检查状态"
        lines += [f"{verdict}：{relative_link(memcheck,target)}。该结果覆盖已运行的功能测试，不替代异步同步协议审查。", ""]
    witness=results/"mixed_precision_witness.json"
    if witness.exists():
        lines += [f"独立精确叶乘法量化反例：{relative_link(witness,target)}。即使叶乘法无额外累加误差，最终 S/T 量化也可能超过固定 floor。", ""]
    ncu=results/"profile_baseline.txt"
    if ncu.exists():
        text=ncu.read_text(encoding="utf-8",errors="replace")
        if "InterprocessLockFailed" in text:
            lines += [f"Nsight Compute probe 失败：`/tmp/nsight-compute-lock` 无法创建/打开，报告 `InterprocessLockFailed`，未获得 kernel 指标。本会话未重新 probe。证据：{relative_link(ncu,target)}。", ""]
    timeline=results/"timeline_lt.txt"
    stages,shape=stage_profile(timeline)
    if stages:
        lines += [f"Nsight Systems 使用 cudaProfilerApi 范围，只捕获调优后的最终完整调用。捕获对象：`{shape or '见日志'}`。证据：{relative_link(timeline,target)}。", "",
                  table_line(["阶段", "GPU kernel 时间占比", "累计 ms", "instances"]),table_line(["---", "---:", "---:", "---:"])]
        for stage in ("leaves","pack","combine"):
            if stage in stages:
                percentage,ns,instances=stages[stage]
                lines.append(table_line([stage,f"{percentage:.1f}%",f"{ns/1e6:.6f}",instances]))
        lines += ["", "阶段比例约为叶乘法 75%、两侧 packing 12%、combine 13%，对应一次被 profiler 捕获的 BF16 8192 调用。比例分母为 kernel 执行时间，不包含所有主机调度间隙；不能将 profiler replay/采样延迟作为 steady-state 性能成绩，也不能将该比例推广至全部尺寸或 FP16。", ""]
    release=results/"hopper_release_resources.txt"
    if release.exists():
        lines += [f"Release 构建资源核验：{relative_link(release,target)}。六个普通 Hopper leaf 核均为 REG 168、STACK/LOCAL 0，SASS LDL/STL 计数均为 0；warpgroup 存在 232/40 的寄存器重新分配。早期 probe 编译日志的 spill 数不代表实际 Release 对象。静态 SHARED=1024 之外还有动态 shared storage，零 spill 本身不证明性能瓶颈已经消除。", ""]
    for path in (results/"environment.txt",results/"dependency.txt"):
        if path.exists():
            lines += [f"环境/依赖记录：{relative_link(path,target)}。", ""]
    if warnings:
        lines += ["## 数据检查提示", ""]
        lines += [f"- {warning}" for warning in warnings]
        lines.append("")
    lines += ["## 后续验收", "",
              "需要同时满足固定数值门槛、完整调用性能和独立功能/内存检查，才能为特定 dtype 与 shape 启用快速路径。SW 的数值失败行、所有 unchecked 行、所有 dense fallback 行均不能用于 SW 性能验收或 SW 认证。fallback 通过只说明普通 GEMM 路由正常。", "",
              "性能目标需要预先指定的至少两个相邻大尺寸，在同一 dtype、执行模式、缓存/轮换合同、算法版本和 variant 下，各自 median speedup ≥ 1.05，并由完整 trials≥100、groups≥5、correct_trials≥3 配置与超出噪声的重复数据支持。各尺寸也须通过冻结的逐试验精度门槛。单行、短测或跨精度/Graph 模式拼接的结果不足以完成验收。若约 5% 的收益与 CV/组分布变化相近，保留原始分组日志并增加独立重复确认；这里不从五组数据声称置信区间，也不自动宣布胜者。", "",
              "最终报告通过 `--files` 选择完整最终数据集；默认含全部轨迹的报告用于实验回顾。后续可独立评估保持原输入 FP32 信息的叶策略、计入时间的补偿/缩放与不同量化合同，任何合同变更必须明确记录并重新建立同精度基线。", ""]
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=Path("results"))
    parser.add_argument("--trajectories",type=Path,default=Path("trajectory"))
    parser.add_argument("--output",type=Path,default=Path("results/REPORT.md"))
    parser.add_argument("--plots-dir",type=Path,default=Path("results/plots"))
    parser.add_argument("--files",type=Path,nargs="*",help="Use only these CSVs instead of discovery")
    parser.add_argument("--no-plots",action="store_true",help="Generate the stdlib Markdown report only")
    args=parser.parse_args()
    paths=args.files if args.files is not None else discover(args.results_dir,args.trajectories)
    paths=[path.resolve() for path in paths if path.is_file()]
    records,warnings=read_records(paths)
    if not records:
        parser.error("no valid CSV records available")
    artifacts=[] if args.no_plots else export_figures(records,args.plots_dir.resolve())
    report(records,warnings,artifacts,args.output.resolve(),args.results_dir.resolve())
    print(f"Wrote {args.output}: {len(records)} rows from {len({r.source for r in records})} CSV files")
    for artifact in artifacts:
        print(f"Wrote {artifact}")
    for warning in warnings:
        print(f"WARNING {warning}")


if __name__ == "__main__":
    main()
