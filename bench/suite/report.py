"""Reproducible suite summaries and static HTML/SVG reports, using stdlib only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
import random
import statistics


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _positive_finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _implementation_ok(run):
    return run.get("status") == "ok" and run.get("implementation_pass") is True


def _quality_gate(run):
    # The native GEMM comparison has no application metric. Its dense baseline
    # uses the finite execution contract; the candidate uses its local FP64 gate.
    if run.get("kind") == "gemm":
        return run.get("implementation_pass") if run.get("backend") == "cublaslt" else run.get("gemm_gate_pass")
    return run.get("application_quality_pass")


def _ratio_statistics(ratios):
    n = len(ratios)
    logs = list(map(math.log, ratios))
    point = math.exp(statistics.mean(logs)) if logs else None
    low = high = None
    if n >= 2:
        rng = random.Random(20261021)
        boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=n))) for _ in range(2000))
        low, high = boot[49], boot[1949]
    return point, low, high


def _confidence_note(n):
    return ("insufficient independent processes" if n == 1 else
            "few independent processes; interval is exploratory" if n == 2 else
            "" if n >= 3 else "no paired measurements")


def process_summary(runs, baseline=None, expected_processes=None):
    """Keep timing evidence and quality-qualified process ratios distinct.

    Baseline records are matched by case, mode and process index. Standalone
    candidate records need explicit baseline status, implementation and quality
    evidence; a previously recorded ``qualified_win`` never substitutes for it.
    """
    grouped = defaultdict(list)
    identities = defaultdict(list)
    for run in runs:
        grouped[(run["case_id"], run["mode"], run["backend"])].append(run)
        identities[(run["case_id"], run["mode"], run["process_index"], run["backend"])].append(run)
    summaries = []
    for (case, mode, backend), rows in sorted(grouped.items()):
        processes = defaultdict(list)
        for row in rows:
            processes[row["process_index"]].append(row)
        # Duplicate archive observations cannot increase n or weight medians.
        # Conflicting observations of one process are excluded, not cherry-picked.
        conflicting = {index for index, items in processes.items()
                       if any(item != items[0] for item in items[1:])}
        observed = [items[0] for items in processes.values()]
        successful = [r for r in observed if r["process_index"] not in conflicting and _implementation_ok(r)]
        expected = expected_processes if expected_processes is not None else len(processes)
        missing = len(set(range(expected)) - set(processes)) if expected_processes is not None else 0
        ratios, qualified_ratios = [], []
        baseline_failed, baseline_unchecked, baseline_missing = 0, 0, 0
        for row in observed:
            selected_baseline = baseline or row.get("baseline_backend")
            if selected_baseline:
                matched = identities.get((case, mode, row["process_index"], selected_baseline), [])
                baseline_present = bool(matched)
                baseline_execution = (bool(matched) and all(_implementation_ok(r) for r in matched) and
                                      all(r == matched[0] for r in matched[1:]))
                gates = [_quality_gate(r) for r in matched]
                baseline_quality = (True if gates and all(g is True for g in gates) else
                                    False if any(g is False for g in gates) else None)
            else:
                baseline_present = any(k in row for k in ("baseline_status", "baseline_implementation_pass", "baseline_quality_pass"))
                baseline_execution = row.get("baseline_status") == "ok" and row.get("baseline_implementation_pass") is True
                baseline_quality = row.get("baseline_quality_pass")
            baseline_failed += baseline_quality is False
            baseline_unchecked += baseline_quality is not True and baseline_quality is not False
            baseline_missing += not baseline_present
            if row not in successful or not _positive_finite(row.get("speedup")):
                continue
            # Old archives can still expose unqualified timing ratios without
            # baseline evidence; qualification always requires that evidence.
            if selected_baseline and not baseline_execution:
                continue
            ratios.append(float(row["speedup"]))
            if _quality_gate(row) is True and baseline_execution and baseline_quality is True:
                qualified_ratios.append(float(row["speedup"]))
        n, qualified_n = len(ratios), len(qualified_ratios)
        point, low, high = _ratio_statistics(ratios)
        qualified_point, qualified_low, qualified_high = _ratio_statistics(qualified_ratios)
        complete = not missing and not conflicting and len(successful) == expected
        all_quality = all(_quality_gate(r) is True for r in observed)
        supported = bool(qualified_n >= 3 and qualified_low is not None and qualified_low > 1 and
                         complete and qualified_n == expected)
        method = "paired independent-process log-ratio percentile bootstrap (2000 resamples)"
        summaries.append({"case_id": case, "mode": mode, "backend": backend,
            "process_count": len(processes), "expected_process_count": expected,
            "missing_process_count": missing, "conflicting_process_count": len(conflicting),
            "successful_process_count": len(successful), "paired_process_count": n,
            "qualified_paired_process_count": qualified_n, "unqualified_timing_pair_count": n - qualified_n,
            "invalid_ratio_process_count": sum(r.get("speedup") is not None and not _positive_finite(r.get("speedup")) for r in successful),
            "missing_ratio_process_count": sum(r.get("speedup") is None for r in successful),
            "speedup_geomean": point, "speedup_ci95_low": low, "speedup_ci95_high": high,
            "speedup_scope": "execution-valid timing; numerical quality may fail or be unchecked",
            "qualified_speedup_geomean": qualified_point,
            "qualified_speedup_ci95_low": qualified_low, "qualified_speedup_ci95_high": qualified_high,
            "qualified_speedup_scope": "candidate and paired baseline passed implementation and applicable quality gates",
            "confidence_method": method if n >= 2 else "insufficient independent processes",
            "qualified_confidence_method": method if qualified_n >= 2 else "insufficient independent processes",
            "statistical_note": _confidence_note(n), "qualified_statistical_note": _confidence_note(qualified_n),
            "statuses": dict(Counter(r.get("status", "missing") for r in rows)),
            "application_quality_pass_count": sum(r.get("application_quality_pass") is True for r in observed),
            "gemm_gate_pass_count": sum(r.get("gemm_gate_pass") is True for r in observed),
            "quality_checked_count": sum(isinstance(_quality_gate(r), bool) for r in observed),
            "quality_failed_process_count": sum(_quality_gate(r) is False for r in observed),
            "quality_unchecked_process_count": sum(not isinstance(_quality_gate(r), bool) for r in observed),
            "baseline_quality_failed_process_count": baseline_failed,
            "baseline_quality_unchecked_process_count": baseline_unchecked,
            "baseline_missing_process_count": baseline_missing,
            "qualified_win_count": sum(ratio > 1 for ratio in qualified_ratios),
            "implementation_pass_count": sum(r.get("implementation_pass") is True for r in observed),
            "implementation_failed_process_count": sum(r.get("implementation_pass") is False for r in observed),
            "implementation_unchecked_process_count": sum(not isinstance(r.get("implementation_pass"), bool) for r in observed),
            "all_observed_quality_pass": all_quality,
            "all_paired_quality_pass": bool(n and qualified_n == n),
            "statistically_supported_timing_speedup": bool(n >= 3 and low is not None and low > 1 and complete and n == expected),
            "statistically_supported_speedup": supported,
            "statistically_supported_qualified_speedup": supported,
            "primary_time_basis": observed[0].get("primary_time_basis", observed[0].get("speedup_time_basis",
                                  "steady_wall_ms" if observed[0].get("kind") != "gemm" else "steady_event_ms")),
            "median_event_ms": _median(r.get("steady_event_ms") for r in successful),
            "median_wall_ms": _median(r.get("steady_wall_ms") for r in successful),
            "median_effective_gflops": _median(r.get("effective_gflops") for r in successful)})
    return summaries


def _median(values):
    values = [x for x in values if _positive_finite(x)]
    return statistics.median(values) if values else None


def _fmt(value):
    if value is None:
        return "unchecked"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _table(rows, fields):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(label)}</th>' for _, label in fields) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(_fmt(row.get(key))) + '</td>'
            for key, _ in fields) + '</tr>' for row in rows) + '</tbody></table>'


def speedup_svg(summary, qualified=False):
    prefix = "qualified_" if qualified else ""
    value_key, low_key, high_key = (prefix + "speedup_" + suffix for suffix in ("geomean", "ci95_low", "ci95_high"))
    count_key = "qualified_paired_process_count" if qualified else "paired_process_count"
    rows = [s for s in summary if s[value_key] is not None]
    if not rows:
        return '<p>No quality-qualified paired performance measurements.</p>' if qualified else '<p>No paired performance measurements.</p>'
    width, left, right, row_h = 1100, 365, 40, 30
    height = 65 + row_h * len(rows)
    max_x = max(1.2, max(s[high_key] or s[value_key] for s in rows) * 1.08)
    scale = lambda x: left + x / max_x * (width - left - right)
    scope = "Quality-qualified speedup" if qualified else "Timing speedup (quality may fail or be unchecked)"
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Speedup with independent process confidence intervals">',
             f'<text x="{left}" y="17">{scope}; 95% process bootstrap interval</text>',
             f'<line x1="{scale(1):.2f}" y1="28" x2="{scale(1):.2f}" y2="{height-20}" stroke="#555" stroke-dasharray="4 3"/>']
    for i, row in enumerate(rows):
        y = 47 + i * row_h
        point = row[value_key]
        color = '#16815d' if qualified or row['all_paired_quality_pass'] else '#b45d14'
        label = f"{row['case_id']} / {row['mode']} / {row['backend']} (n={row[count_key]})"
        parts.append(f'<text x="5" y="{y+4}" font-size="11">{html.escape(label)}</text>')
        if row[low_key] is not None:
            parts.append(f'<line x1="{scale(row[low_key]):.2f}" y1="{y}" x2="{scale(row[high_key]):.2f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{scale(point):.2f}" cy="{y}" r="4" fill="{color}"/><text x="{scale(point)+7:.2f}" y="{y-6}" font-size="11">{point:.3f}×</text>')
    parts.append('</svg>')
    return ''.join(parts)


def runtime_svg(summary, config):
    cases = {c['id']: c for c in config.get('cases', [])}
    series = defaultdict(list)
    for row in summary:
        case = cases.get(row['case_id'], {})
        n = case.get('n', case.get('hidden'))
        kind = case.get('app', case.get('application', case.get('kind')))
        basis = row.get('primary_time_basis', 'steady_event_ms' if kind == 'gemm' else 'steady_wall_ms')
        ms = row['median_event_ms'] if basis == 'steady_event_ms' else row['median_wall_ms']
        if ms is None:
            ms = row['median_wall_ms'] if basis == 'steady_event_ms' else row['median_event_ms']
        if n and ms is not None and ms > 0:
            # Only connect dimension sweeps with the same algorithm, precision,
            # stopping rule, spectrum and shape ratios. In particular, never
            # connect sqrt to inverse_sqrt or fixed work to time-to-quality.
            dimensions = {'n', 'm', 'k', 'tokens', 'hidden', 'intermediate'}
            signature = {k: v for k, v in case.items() if k not in dimensions | {'id', 'description', 'asset_description'}}
            signature.setdefault('evaluation', 'fixed_work')
            signature['shape_ratios'] = {key: case[key] / n for key in dimensions if key in case}
            if 'occupied' in signature:
                signature['occupied_fraction'] = signature.pop('occupied') / n
            if 'nocc' in signature:
                signature['occupied_fraction'] = signature.pop('nocc') / n
            key = (case.get('kind', case.get('app', case.get('application'))), row['backend'], row['mode'],
                   case.get('dtype', 'fp32'), json.dumps(signature, sort_keys=True))
            series[key].append((n, ms))
    series = {k: sorted(v) for k, v in series.items() if len(v) >= 2}
    if not series:
        return '<p>Scale curves require at least two dimensions for the same application, backend, mode, and dtype.</p>'
    xs = [math.log2(n) for points in series.values() for n, _ in points]
    ys = [math.log10(ms) for points in series.values() for _, ms in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x = lambda n: 90 + (math.log2(n)-x0) / max(x1-x0, .01) * 650
    y = lambda ms: 330 - (math.log10(ms)-y0) / max(y1-y0, .01) * 280
    colors = ['#2065a8', '#16815d', '#b45d14', '#b33154', '#7346a3', '#187f91']
    out = ['<svg viewBox="0 0 1100 380" role="img" aria-label="Application runtime scaling">',
           '<text x="70" y="20">Median runtime (log scale), dimension (log scale)</text>']
    for i, (key, points) in enumerate(sorted(series.items())):
        color = colors[i % len(colors)]
        coords = ' '.join(f'{x(n):.2f},{y(ms):.2f}' for n, ms in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for n, ms in points:
            out.append(f'<circle cx="{x(n):.2f}" cy="{y(ms):.2f}" r="3" fill="{color}"><title>n={n}, {ms:.5g} ms</title></circle>')
        label = " / ".join(map(str, key[:4]))
        out.append(f'<text x="765" y="{40+i*18}" fill="{color}" font-size="10"><title>{html.escape(key[4])}</title>{html.escape(label)}</text>')
    for val in sorted(set(n for points in series.values() for n, _ in points)):
        out.append(f'<text x="{x(val):.2f}" y="355" font-size="11">{val}</text>')
    for val in (min(ms for points in series.values() for _, ms in points), max(ms for points in series.values() for _, ms in points)):
        out.append(f'<text x="5" y="{y(val):.2f}" font-size="11">{val:.4g} ms</text>')
    return ''.join(out) + '</svg>'


def build_report(directory):
    directory = Path(directory)
    runs, samples, metrics = [read_jsonl(directory / f'{name}.jsonl') for name in ('runs', 'samples', 'metrics')]
    config_path = directory / 'resolved_config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    summary = process_summary(runs, baseline=config.get('baseline'),
                              expected_processes=config.get('measurement', {}).get('independent_processes'))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    csv_rows = [{**s, 'statuses': json.dumps(s['statuses'], sort_keys=True)} for s in summary]
    with (directory / 'summary.csv').open('w', newline='') as stream:
        fields = list(csv_rows[0]) if csv_rows else ['case_id', 'mode', 'backend']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    status_counts = dict(Counter(r.get('status', 'missing') for r in runs))
    summary_fields = [('case_id', 'Case'), ('mode', 'Mode'), ('backend', 'Backend'), ('statuses', 'All statuses'),
                      ('missing_process_count', 'Missing processes'), ('paired_process_count', 'Timing pairs'),
                      ('speedup_geomean', 'Timing speedup'), ('qualified_paired_process_count', 'Qualified pairs'),
                      ('qualified_speedup_geomean', 'Qualified speedup'),
                      ('qualified_speedup_ci95_low', 'Qualified CI low'), ('qualified_speedup_ci95_high', 'Qualified CI high'),
                      ('quality_failed_process_count', 'Quality failed'), ('quality_unchecked_process_count', 'Quality unchecked'),
                      ('baseline_quality_failed_process_count', 'Baseline quality failed'),
                      ('baseline_quality_unchecked_process_count', 'Baseline quality unchecked'),
                      ('statistically_supported_speedup', 'Supported qualified speedup'),
                      ('qualified_win_count', 'Qualified wins'), ('qualified_statistical_note', 'Qualified statistical note')]
    run_fields = [('case_id', 'Case'), ('mode', 'Mode'), ('process_index', 'Process'), ('backend', 'Backend'),
                  ('status', 'Status'), ('implementation_pass', 'Implementation'), ('gemm_gate_pass', 'Local GEMM'),
                  ('application_quality_pass', 'Application quality'), ('performance_win', 'Faster'),
                  ('steady_event_ms', 'Event ms'), ('steady_wall_ms', 'Wall ms'),
                  ('cold_setup_ms', 'Cold setup ms'), ('break_even_calls', 'Break even calls'), ('error', 'Error')]
    has_linpack = any(r.get('kind') == 'linpack' for r in runs)
    has_throughput = any(s['median_effective_gflops'] is not None for s in summary)
    throughput_note = ("Equivalent FP32 throughput uses synchronized complete-application wall time and each application's "
                       "declared classical work convention, rather than SW executed instructions.")
    linpack_note = ("LINPACK-style FP32 solve rows use equivalent classical solve FLOPs. They are not FP64 HPL Rmax.")
    if has_throughput:
        summary_fields.append(('median_effective_gflops', 'Equivalent FP32 GFLOPS (wall)'))
        run_fields.append(('effective_gflops', 'Equivalent FP32 GFLOPS (wall)'))
    metric_rows = []
    for row in metrics:
        values = row.get('values', {})
        for metric_name, value in values.items():
            if isinstance(value, (float, int, bool)) or value is None:
                metric_rows.append({**{k: row.get(k) for k in ('case_id', 'mode', 'backend', 'process_index', 'trial')},
                                    'metric': metric_name, 'value': value})
    metrics_table = _table(metric_rows, [('case_id', 'Case'), ('backend', 'Backend'), ('mode', 'Mode'),
        ('process_index', 'Process'), ('trial', 'Trial'), ('metric', 'Metric'), ('value', 'Value')])
    title = f"MegaFMM suite: {config.get('suite', directory.name)}"
    document = '<!doctype html><html><head><meta charset="utf-8"><title>' + html.escape(title) + '</title>' + \
        '<style>body{font:14px system-ui,sans-serif;margin:30px;color:#162030}table{border-collapse:collapse;display:block;overflow:auto;margin:18px 0}th,td{border:1px solid #d5dce3;padding:7px;text-align:left;white-space:nowrap}th{background:#edf2f7}svg{width:100%;max-width:1300px;border:1px solid #dde3ea;margin:12px 0}p{max-width:1100px;line-height:1.6}code{background:#f0f3f7;padding:2px 4px}</style></head><body><h1>' + \
        html.escape(title) + '</h1><p>' + html.escape(f"{len(runs)} backend process records, {len(samples)} raw group samples; statuses: {status_counts}") + '</p>' + \
        '<p>Each confidence interval resamples independent process pairs. Timer groups remain correlated observations within a process. A single process provides no confidence interval; two processes are exploratory. Qualified ratios require both the candidate and the paired baseline to pass execution, implementation and applicable quality gates. A supported qualified speedup additionally requires at least three qualified pairs, a confidence interval above one and complete process coverage. Separate timing ratios may include rejected or unchecked numerical quality. Runtime medians exclude failed, conflicting or implementation-rejected processes. All failures, unsupported cases, missing external assets and OOM records remain visible. Green timing points have qualified pairs throughout; orange timing points include unqualified evidence. Synthetic algorithm tests do not establish model training quality or chemical energy accuracy.</p>' + \
        '<p><a href="manifest.json">Manifest</a> · <a href="resolved_config.json">Resolved config</a> · <a href="summary.csv">CSV</a> · <a href="runs.jsonl">Runs</a> · <a href="samples.jsonl">Raw samples</a> · <a href="metrics.jsonl">Quality metrics</a></p>' + \
        '<h2>Quality-qualified paired process speedup</h2>' + speedup_svg(summary, qualified=True) + _table(summary, summary_fields) + \
        '<h2>Timing evidence before quality qualification</h2>' + speedup_svg(summary) + \
        '<h2>Runtime scaling</h2>' + runtime_svg(summary, config) + \
        ('<p>' + throughput_note + (' ' + linpack_note if has_linpack else '') + '</p>' if has_throughput else '') + \
        '<h2>Every backend process record</h2>' + _table(runs, run_fields) + \
        '<h2>Quality observations</h2>' + metrics_table + '</body></html>'
    (directory / 'report.html').write_text(document)
    lines = [f'# {title}', '', f'Backend process statuses: `{status_counts}`.', '',
        'Confidence intervals use independent process pairs, not timing groups. Qualified pairs require candidate and baseline execution, implementation and quality gates. Timing ratios may include failed or unchecked numerical quality; failed execution times are excluded from medians.', '',
        '| Case | Mode | Backend | Timing pairs | Timing speedup | Qualified pairs | Qualified speedup | Qualified 95% interval | Missing processes | Quality failed / unchecked | Statuses |',
        '|---|---|---|---:|---:|---:|---:|---|---:|---|---|']
    for s in summary:
        interval = f"{_fmt(s['qualified_speedup_ci95_low'])} – {_fmt(s['qualified_speedup_ci95_high'])}" if s['qualified_paired_process_count'] >= 2 else 'insufficient independent processes'
        lines.append(f"| {s['case_id']} | {s['mode']} | {s['backend']} | {s['paired_process_count']} | {_fmt(s['speedup_geomean'])} | {s['qualified_paired_process_count']} | {_fmt(s['qualified_speedup_geomean'])} | {interval} | {s['missing_process_count']} | {s['quality_failed_process_count']} / {s['quality_unchecked_process_count']} | {_fmt(s['statuses'])} |")
    throughput_rows = [s for s in summary if s['median_effective_gflops'] is not None]
    if throughput_rows:
        lines.extend(['', throughput_note])
        if has_linpack:
            lines.extend(['', linpack_note])
        lines.extend(['',
                      '| Case | Backend | Median wall ms | Equivalent FP32 GFLOPS |',
                      '|---|---|---:|---:|'])
        for s in throughput_rows:
            lines.append(f"| {s['case_id']} | {s['backend']} | {_fmt(s['median_wall_ms'])} | {_fmt(s['median_effective_gflops'])} |")
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args(argv)
    try:
        build_report(args.directory)
    except (ValueError, OSError) as exc:
        p.exit(2, f'report error: {exc}\n')
    print(args.directory / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
