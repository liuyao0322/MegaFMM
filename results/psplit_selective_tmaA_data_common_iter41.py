#!/usr/bin/env python3
"""Pure CPU archive mathematics; no iteration-specific routes, sizes or resource forecasts.

CSV strings remain authoritative. Functions verify group statistics against the
closed stdout values with their printed precision; they never replace CSV data.
This module performs no I/O, imports no historical main and executes no GPU code.
"""
from __future__ import annotations
from decimal import Decimal
import math
import re

TIME_RE = re.compile(r'^TIME shape=(\S+) dtype=(\S+) group=(\d+) case=(\S+) event_ms=(\S+) wall_ms=(\S+)$', re.M)
CHECK_RE = re.compile(r'^CHECK (\S+) (\S+) trial=(\d+) dense_eF=(\S+) solution_eF=(\S+) gate=(\S+) pass=([01])$', re.M)
TUNE_RE = re.compile(r'^TUNE shape=(\S+) dtype=(\S+) case=(\S+) config=(.*)$', re.M)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def printed_error(token):
    value = Decimal(str(token))
    require(value.is_finite(), 'Nonfinite printed token has no finite rounding interval')
    return float(Decimal(10) ** value.as_tuple().exponent) / 2


def quantile(values, q):
    values = sorted(values)
    require(values and 0 <= q <= 1, 'A quantile requires samples and probability in[0,1]')
    position = q * (len(values) - 1)
    low = int(position); high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def statistics(values):
    require(values and all(math.isfinite(x) and x > 0 for x in values), 'Positive finite timing groups required')
    mean = sum(values) / len(values)
    cv = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)) / mean
    return quantile(values, .5), quantile(values, .1), quantile(values, .9), cv


def records(text):
    times = [dict(zip(('shape','dtype','group','case','event_ms','wall_ms'), match)) for match in TIME_RE.findall(text)]
    checks = [dict(zip(('shape','dtype','trial','dense_ef','solution_ef','gate_limit','pass'), match)) for match in CHECK_RE.findall(text)]
    tunes = [dict(zip(('shape','dtype','case','config'), match)) for match in TUNE_RE.findall(text)]
    return times, checks, tunes


def verify_timing_row(row, times, tunes, groups):
    shape = ','.join(row[k] for k in ('m','n','k'))
    own = [t for t in times if (t['shape'],t['dtype']) == (shape,row['dtype'])]
    require(len(own) == 2 * groups and {(t['case'],int(t['group'])) for t in own} ==
            {(case,g) for case in ('dense','solution') for g in range(groups)}, 'Wrong complete group-mean TIME domain')
    require(row['quantile_unit'] == 'group_mean', 'CSV quantiles must describe group means')
    summaries = {}
    for case in ('dense','solution'):
        samples = sorted((t for t in own if t['case'] == case), key=lambda t:int(t['group']))
        events = [float(t['event_ms']) for t in samples]; walls = [float(t['wall_ms']) for t in samples]
        require(all(math.isfinite(x) and x > 0 for x in walls), 'Positive finite synchronized batchwall groups required')
        stats = statistics(events)
        event_error = max(printed_error(t['event_ms']) for t in samples)
        wall_error = max(printed_error(t['wall_ms']) for t in samples)
        for expected, field in zip(stats[:3], (case+'_event_ms',case+'_p10_ms',case+'_p90_ms')):
            require(abs(expected - float(row[field])) <= event_error + printed_error(row[field]) + 1e-12,
                    'CSV '+field+' outside printed group interval')
        cv_bound = 3 * event_error / min(events) + stats[3] * event_error / min(events) + printed_error(row[case+'_cv']) + 1e-11
        require(abs(stats[3] - float(row[case+'_cv'])) <= cv_bound, 'CSV populationCV outside printed group interval')
        require(abs(quantile(walls,.5) - float(row[case+'_wall_ms'])) <= wall_error + printed_error(row[case+'_wall_ms']) + 1e-12,
                'CSV batchwall median outside printed group interval')
        configs = [t['config'] for t in tunes if (t['shape'],t['dtype'],t['case']) == (shape,row['dtype'],case)]
        require(configs == [row[case+'_lt_config']], 'CSV '+case+' config differs from single bound TUNE record')
        summaries[case] = {'event_ms_group_means':[t['event_ms'] for t in samples],
            'batchwall_ms_per_call_group_means':[t['wall_ms'] for t in samples],
            'recomputed_group_mean_median':stats[0], 'recomputed_group_mean_p10':stats[1],
            'recomputed_group_mean_p90':stats[2], 'recomputed_populationCV':stats[3],
            'all_declared_groups_retained':True}
    return own, summaries


def verify_dense_binding(row, workspace_limit):
    config = row['dense_lt_config']
    require(config.startswith('dense:') and ';tuned=1;tune=4w/3x10median;' in config,
            'Require independently tuned same-row Dense config before capture')
    bound = re.search(r';bound_candidate=(\d+)/(\d+);tuned_candidate=(\d+);bound_workspace=(\d+);',config)
    require(bound is not None and 0 <= int(bound[1]) < int(bound[2]) and bound[1] == bound[3] and
            int(bound[4]) <= int(row['dense_workspace']) <= workspace_limit,
            'Dense tuning candidate/workspace does not match current bound call')


def verify_fresh_row(row, checks, correct_trials, floor):
    shape = ','.join(row[k] for k in ('m','n','k'))
    own = [c for c in checks if (c['shape'],c['dtype']) == (shape,row['dtype'])]
    require(len(own) == correct_trials and {int(c['trial']) for c in own} == set(range(correct_trials)), 'Wrong current fresh CHECK domain')
    require(float(row['gate_floor']) == floor and row['numerical_pass'] in ('0','1') and int(row['nonfinite']) >= 0,
            'Original numerical floor/NF/aggregate flag changed')
    for check in own:
        dense, solution, gate = map(float,(check['dense_ef'],check['solution_ef'],check['gate_limit']))
        require(math.isfinite(dense) and math.isfinite(gate), 'Dense CHECK error/gate must be finite')
        require(abs(gate-(3*dense+floor)) <= printed_error(check['gate_limit'])+3*printed_error(check['dense_ef'])+1e-12,
                'Original per-trial3Dense+floor gate formula changed')
        if row['nonfinite'] == '0':
            require(math.isfinite(solution) and (check['pass']=='1') == (solution <= gate), 'Fresh CHECK flag disagrees with original gate')
    passed = all(c['pass']=='1' for c in own) and int(row['nonfinite']) == 0
    require((row['numerical_pass']=='1') == passed, 'CSV fresh aggregate must retain original AND/NF result')
    for field, raw_field in (('dense_ef','dense_ef'),('solution_ef','solution_ef')):
        values = [float(c[raw_field]) for c in own]
        if all(math.isfinite(x) for x in values) and math.isfinite(float(row[field])):
            require(abs(max(values)-float(row[field])) <= max(printed_error(c[raw_field]) for c in own)+printed_error(row[field])+1e-12,
                    'CSV worst fresh error differs from stdout CHECK values')
    return own, passed
