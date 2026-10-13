#!/usr/bin/env python3
"""Owner-run iteration39 archive generator; default output is stdout.

Only actual full-protocol raw rows are copied. The independent audit separately
recomputes statistics, gates, Git bindings and the contents of compiled evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

TIME_RE = re.compile(r'^TIME shape=([^ ]+) dtype=([^ ]+) group=(\d+) case=([^ ]+) event_ms=([^ ]+) wall_ms=([^ ]+)$', re.M)
CHECK_RE = re.compile(r'^CHECK ([^ ]+) ([^ ]+) trial=(\d+) dense_eF=([^ ]+) solution_eF=([^ ]+) gate=([^ ]+) pass=(\d+)$', re.M)
CONTROL_ACTUAL = {('mega', '36'): 'megakernel_wgmma_psplit_64x32',
                  ('mega', '39'): 'megakernel_wgmma_psplit_rsa_64x32',
                  ('mega', '40'): 'megakernel_wgmma_psplit_rsa_bk32_64x32',
                  ('lt', '1'): 'winograd_lt_batched'}
PLAN_DEFAULT = 'results/psplit_bn161024_plan_iter39.json'
IDENTITY_DEFAULT = 'results/psplit_bn16_measured_identity_iter39.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row['requested'], str(row['variant']), row['dtype'], row['mode']


def key_object(value):
    return dict(zip(('algorithm', 'variant', 'dtype', 'mode'), value))


def us(row, field='solution_event_ms'):
    return float(row[field]) * 1000


def read_exclusions(path, expected, plan_binding, identity):
    if path is None:
        return {}, None
    obj = json.loads(Path(path).read_text())
    require(obj.get('schema') == 'megafmm.iter39.measurement-exclusions.v1',
            'Partial scope requires explicit iteration39 exclusions schema')
    require(obj.get('measurement_source_commit') == identity['measurement_source_commit'],
            'Exclusions measured-source mismatch')
    require(obj.get('frozen_sources') == identity['frozen_sources'] and
            obj.get('frozen_binaries') == identity['frozen_binaries'],
            'Exclusions source/binary freeze mismatch')
    require(obj.get('predeclared_plan') == plan_binding, 'Exclusions frozen-plan mismatch')
    excluded = {}
    for item in obj.get('unmeasured_rows', []):
        value = item['algorithm'], str(item['variant']), item['dtype'], item['mode']
        require(value in expected and value not in excluded and value[:2] in {('mega', '42'), ('mega', '43')},
                'Only unique new42/new43 rows may be excluded')
        require(item.get('classification') == 'compiled_safety_blocked',
                'Only compiled-safety blocks authorize missing rows')
        require(item.get('reason') and item.get('evidence_path') and item.get('evidence_sha256'),
                'Each missing row requires a reason and actual compiled gate binding')
        require(sha(item['evidence_path']) == item['evidence_sha256'], 'Compiled gate evidence SHA mismatch')
        gate = identity['independent_compiled_gate']
        require((item['evidence_path'], item['evidence_sha256']) == (gate['path'], gate['sha256']),
                'Excluded rows must bind the latest measured-identity compiled gate, not an initial or retained gate')
        excluded[value] = item
    blocked = {str(v) for v in (42, 43) if identity['route_configurations'][str(v)]['enabled'] is False} if 'route_configurations' in identity else None
    allowed = {value for value in expected if value[:2] in {('mega', '42'), ('mega', '43')}}
    require(set(excluded) <= allowed and len(excluded) in (4, 8),
            'Iteration39 exclusions must name complete blocked variants, four keys each')
    for variant in (42, 43):
        own = {value for value in allowed if value[1] == str(variant)}
        require(not set(excluded) & own or own <= set(excluded),
                'A blocked variant must exclude all four dtype/mode keys')
    return excluded, {'path': str(path), 'sha256': sha(path), 'rows': obj['unmeasured_rows'],
                      'measurement_source_commit': obj['measurement_source_commit'],
                      'frozen_sources': obj['frozen_sources'], 'frozen_binaries': obj['frozen_binaries'],
                      'predeclared_plan': obj['predeclared_plan'],
                      'scope': 'Manifest identity and onsite gate SHA are checked; independent audit verifies gate contents and latest applicable compile revision.'}


def comparison(mode, dtype, baseline, candidate):
    old, new = baseline['csv_row'], candidate['csv_row']
    a, b = us(old), us(new)
    return {'mode': mode, 'dtype': dtype, 'axis': f"{new['variant']}_vs{old['variant']}",
            'baseline_path': baseline['path'], 'baseline_line': baseline['line'],
            'baseline_algorithm': old['requested'],
            'baseline_variant': int(old['variant']), 'baseline_us': a,
            'candidate_path': candidate['path'], 'candidate_line': candidate['line'],
            'candidate_algorithm': new['requested'],
            'candidate_variant': int(new['variant']), 'candidate_us': b,
            'baseline_dense_us': us(old, 'dense_event_ms'),
            'candidate_dense_us': us(new, 'dense_event_ms'),
            'baseline_batch_wall_us': us(old, 'solution_wall_ms'),
            'baseline_dense_batch_wall_us': us(old, 'dense_wall_ms'),
            'candidate_batch_wall_us': us(new, 'solution_wall_ms'),
            'candidate_dense_batch_wall_us': us(new, 'dense_wall_ms'),
            'baseline_over_candidate_own_latency': a / b,
            'own_latency_decrease_percent': (1 - b / a) * 100,
            'baseline_same_row_dense_over_sw': float(old['speedup']),
            'candidate_same_row_dense_over_sw': float(new['speedup']),
            'candidate_numerical_pass': new['numerical_pass'] == '1',
            'baseline_numerical_pass': old['numerical_pass'] == '1',
            'same_binary': True}


def generate(plan_path, identity_path, exclusions_path=None):
    plan = json.loads(Path(plan_path).read_text())
    identity = json.loads(Path(identity_path).read_text())
    require(len(plan['frozen_sources']) == 8 and len(plan['frozen_binaries']) == 2,
            'Require eight-source/two-binary final freeze')
    require(plan['frozen_sources'] == identity['frozen_sources'] and
            plan['frozen_binaries'] == identity['frozen_binaries'], 'Plan/identity freeze mismatch')
    for path, wanted in {**plan['frozen_sources'], **plan['frozen_binaries']}.items():
        require(sha(path) == wanted, 'Frozen source/binary onsite SHA mismatch: ' + path)
    plan_binding = {'path': str(plan_path), 'sha256': sha(plan_path)}
    declared_plan_hashes = []
    if identity.get('predeclared_plan_sha256'):
        declared_plan_hashes.append(identity['predeclared_plan_sha256'])
    for name in ('plan', 'predeclared_plan'):
        declaration = identity.get(name)
        if isinstance(declaration, dict) and declaration.get('sha256'):
            declared_plan_hashes.append(declaration['sha256'])
    require(declared_plan_hashes and all(value == plan_binding['sha256'] for value in declared_plan_hashes),
            'Every declared identity plan SHA must match the frozen measured plan')
    require((plan['warmup'], plan['trials'], plan['groups'], plan['correct_trials'],
             plan['rotate_inputs'], plan['heuristic_candidates'], plan['workspace_limit']) ==
            (20, 100, 5, 3, 1, 32, 8589934592), 'Changed mandatory full timing contract')
    require(plan['shapes'] == [[1024, 1024, 1024]], 'Wrong mandatory shape')
    require(len(plan['dtypes']) == 2 and set(plan['dtypes']) == {'bf16', 'fp16'} and
            len(plan['modes']) == 2 and set(plan['modes']) == {'launch', 'graph'} and
            len(plan['routes']) == 6 and len({(r['algorithm'], str(r['variant'])) for r in plan['routes']}) == 6,
            'Exactly two unique dtypes, two unique modes and six unique routes are required')
    expected = {(r['algorithm'], str(r['variant']), dt, mode)
                for r in plan['routes'] for dt in plan['dtypes'] for mode in plan['modes']}
    target = {(algorithm, variant, dt, mode) for algorithm, variant in
              [('mega', '42'), ('mega', '43'), ('mega', '39'), ('mega', '40'), ('mega', '36'), ('lt', '1')]
              for dt in ('bf16', 'fp16') for mode in ('launch', 'graph')}
    require(expected == target and plan.get('expected_row_count') == 24 and
            len(plan['csv_paths']) == 12 and len(set(plan['csv_paths'])) == 12,
            'Plan must declare exactly42/43/39/40/36/Lt1 full24 and twelve unique CSV paths')
    configurations = plan.get('route_configurations', {})
    for variant in (42, 43):
        require(str(variant) in configurations, 'Both new route configurations are required')
        final = configurations[str(variant)]
        require(isinstance(final.get('enabled'), bool), 'Final enabled status must explicitly follow the fresh compiled gate')
        require(final.get('actual_name') and final.get('producer_quota') is not None and
                len(final.get('consumer_quotas', [])) == 2 and final.get('launch_bounds_min_ctas') is not None,
                'Both final actual names, all role quotas and launch bounds must be declared')
        require(all(final.get(k) is not None for k in ('BM', 'BN', 'BK', 'threads')) and
                all(k in final for k in ('dynamic_shared_bytes', 'static_shared_bytes')),
                'Both final geometries and actual shared fields must be explicitly declared')
        require(not final['enabled'] or all(final[k] is not None for k in ('dynamic_shared_bytes', 'static_shared_bytes')),
                'An enabled measured route requires actual shared allocation; blocked unknown scope remains null')
    require(all(str(v) in configurations and configurations[str(v)].get('actual_name') and
                configurations[str(v)].get('BK') is not None and
                configurations[str(v)].get('producer_quota') is not None and
                len(configurations[str(v)].get('consumer_quotas', [])) == 2 and
                configurations[str(v)].get('launch_bounds_min_ctas') is not None for v in (36, 39, 40)),
            'All three retained controls require frozen actual name, BK, all role quotas and launch bounds')
    latest_gate = identity.get('independent_compiled_gate')
    require(isinstance(latest_gate, dict) and all(latest_gate.get(k) for k in ('path', 'sha256', 'status')),
            'Actual latest independent compiled gate must be bound by measured identity')
    require(sha(latest_gate['path']) == latest_gate['sha256'] and
            not any(token in str(latest_gate['status']).upper() for token in ('PENDING', 'UNKNOWN')),
            'Latest compiled gate bytes/status must be actual, not an initial pending forecast')
    original_gate = json.loads(Path(latest_gate['path']).read_text())
    require('iter39' in original_gate.get('schema', '') and 'compiled' in original_gate.get('schema', '') and
            original_gate.get('status') == latest_gate['status'] and
            original_gate.get('compiled_gate_inherited_from_iteration38') is False,
            'Original iteration39 compiled gate must match status and explicitly reject inherited iteration38 certification')
    actual_names = dict(CONTROL_ACTUAL)
    for variant, config in configurations.items():
        if str(variant) in ('42', '43', '36', '39', '40'):
            actual_names['mega', str(variant)] = config['actual_name']
    excluded, exclusion_binding = read_exclusions(exclusions_path, expected, plan_binding, identity)
    blocked_rows = {value for value in expected if value[0] == 'mega' and value[1] in ('42', '43')
                    and not configurations[value[1]]['enabled']}
    require(set(excluded) == blocked_rows,
            'Final enabled statuses and exact compiled-blocked exclusions disagree')
    runs, rows, seen, missing_files = [], [], set(), []
    for filename in plan['csv_paths']:
        path = Path(filename)
        if not path.exists():
            missing_files.append(filename)
            continue
        with path.open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        require(csv_rows, 'Existing CSV has no result rows: ' + filename)
        log_path, exit_path = path.with_suffix('.log'), path.with_suffix('.exit')
        text = log_path.read_text()
        timing = [dict(zip(('shape', 'dtype', 'group', 'case', 'event_ms', 'wall_ms'), match))
                  for match in TIME_RE.findall(text)]
        checks = [dict(zip(('shape', 'dtype', 'trial', 'dense_ef', 'solution_ef', 'gate_limit', 'pass'), match))
                  for match in CHECK_RE.findall(text)]
        require(len(timing) == len(csv_rows) * 10 and len(checks) == len(csv_rows) * 3,
                'Incomplete TIME/CHECK archive: ' + filename)
        run_pass = True
        for line, row in enumerate(csv_rows, 2):
            value = key(row)
            require(value in expected and value not in seen and value not in excluded,
                    'Unexpected, duplicate or excluded measured row: ' + filename)
            require((row['m'], row['n'], row['k']) == ('1024',) * 3, 'Wrong full shape')
            require((row['trials'], row['groups'], row['correct_trials'], row['quantile_unit'],
                     row['rotate_inputs'], row['workspace_limit']) ==
                    ('100', '5', '3', 'group_mean', '1', '8589934592'), 'Wrong row protocol')
            require(row['cache'] == 'warm_cache' and row['input_working_set_bytes'] == '4194304',
                    'Wrong full warm-cache input working set')
            require(row['actual'] == actual_names[value[:2]], 'Wrong frozen actual route')
            if row['requested'] == 'mega':
                require(row['solution_workspace'] == '0', 'Native Mega has matrix workspace')
            own_times = [t for t in timing if t['dtype'] == row['dtype']]
            own_checks = [c for c in checks if c['dtype'] == row['dtype']]
            require(len(own_times) == 10 and {(t['case'], int(t['group'])) for t in own_times} ==
                    {(case, group) for case in ('dense', 'solution') for group in range(5)}, 'Wrong TIME domain')
            require(len(own_checks) == 3 and {int(c['trial']) for c in own_checks} == {0, 1, 2}, 'Wrong CHECK domain')
            require(all(t['shape'] == '1024,1024,1024' for t in own_times + own_checks), 'Wrong log shape')
            for field in ('dense_event_ms', 'solution_event_ms', 'speedup', 'equivalent_tflops',
                          'dense_wall_ms', 'solution_wall_ms'):
                require(math.isfinite(float(row[field])) and float(row[field]) > 0, 'Invalid timing/performance metric')
            require(int(row['nonfinite']) >= 0 and row['numerical_pass'] in ('0', '1') and
                    all(c['pass'] in ('0', '1') for c in own_checks), 'Invalid gate/NF flags')
            passed = all(c['pass'] == '1' for c in own_checks) and int(row['nonfinite']) == 0
            require((row['numerical_pass'] == '1') == passed, 'Numerical row aggregate mismatch')
            run_pass &= passed
            seen.add(value)
            rows.append({'path': filename, 'line': line, 'csv_row': row,
                         'raw_groups': own_times, 'fresh_checks': own_checks})
        code = int(exit_path.read_text())
        require(code == (0 if run_pass else 2), 'Terminal exit and actual numerical result disagree')
        runs.append({'path': filename, 'sha256': sha(path), 'log_path': str(log_path),
                     'log_sha256': sha(log_path), 'exit_path': str(exit_path),
                     'exit_sha256': sha(exit_path), 'exit': code, 'rows': csv_rows})
    missing = expected - seen
    require(missing == set(excluded), 'Missing rows require exact new42/new43 exclusions')
    require(seen, 'No measured rows')
    pairs = ((42, 39, 'RAW_BN16_42_vs_PRIOR_BK64_39'),
             (42, 40, 'RAW_BN16_42_vs_RAW_BN32_40'),
             (42, 43, 'RAW_BN16_42_vs_SS_BN16_43'),
             (43, 39, 'SS_BN16_43_vs_PRIOR_BK64_39'),
             (43, 36, 'SS_BN16_43_vs_SS_BN32_36'))
    each_new, unavailable, winners, native_winners = [], [], [], []
    axes = {axis: [] for _, _, axis in pairs}
    for mode in plan['modes']:
        for dt in plan['dtypes']:
            group = [z for z in rows if z['csv_row']['mode'] == mode and z['csv_row']['dtype'] == dt]
            mega = {int(z['csv_row']['variant']): z for z in group if z['csv_row']['requested'] == 'mega'}
            for candidate, baseline, axis in pairs:
                if candidate in mega and baseline in mega:
                    record = comparison(mode, dt, mega[baseline], mega[candidate])
                    record.update(axis=axis, controlled_axis_claimed=False,
                        controlled_axis_scope='Observed same-binary whole-call comparison. BN, role quotas, launch bounds and tail layout may change together; no isolated cause, dynamic residency, stall or traffic claim.')
                    each_new.append(record)
                    axis_record = dict(record)
                    for name, variant in (('candidate', candidate), ('baseline', baseline)):
                        conf = configurations[str(variant)]
                        axis_record.update({name+'_'+key: conf[key] for key in ('BM', 'BN', 'BK', 'producer_quota', 'consumer_quotas', 'launch_bounds_min_ctas')})
                    axes[axis].append(axis_record)
                else:
                    unavailable.append({'mode': mode, 'dtype': dt, 'axis': axis,
                        'candidate_variant': candidate, 'baseline_variant': baseline,
                        'missing_keys': [key_object(('mega', str(v), dt, mode)) for v in (candidate, baseline) if v not in mega]})
            for candidates, destination in ((group, winners), (list(mega.values()), native_winners)):
                if candidates:
                    winner = min(candidates, key=lambda z: us(z['csv_row']))
                    row = winner['csv_row']
                    destination.append({'mode': mode, 'dtype': dt, 'path': winner['path'], 'line': winner['line'],
                        'requested': row['requested'], 'actual': row['actual'], 'variant': int(row['variant']),
                        'sw_us': us(row), 'dense_us': us(row, 'dense_event_ms'),
                        'same_row_dense_over_sw': float(row['speedup']), 'numerical_pass': row['numerical_pass'] == '1'})
    comparisons39 = [z for z in each_new if z['baseline_variant'] == 39]
    comparisons40 = [z for z in each_new if (z['candidate_variant'], z['baseline_variant']) == (42, 40)]
    comparisons36 = [z for z in each_new if (z['candidate_variant'], z['baseline_variant']) == (43, 36)]
    comparisons43 = [z for z in each_new if (z['candidate_variant'], z['baseline_variant']) == (42, 43)]
    fresh = [c for z in rows for c in z['fresh_checks']]
    noisy = []
    for z in rows:
        row = z['csv_row']
        if float(row['dense_cv']) >= .02:
            dense_groups = sorted((t for t in z['raw_groups'] if t['case'] == 'dense'), key=lambda t: int(t['group']))
            noisy.append({'path': z['path'], 'line': z['line'], 'row_key': key_object(key(row)),
                          'dense_cv': float(row['dense_cv']), 'solution_cv': float(row['solution_cv']),
                          'dense_event_group_means_us': [float(t['event_ms']) * 1000 for t in dense_groups],
                          'all_five_groups_retained': True})
    measured_native = sorted({int(z['csv_row']['variant']) for z in rows
                              if z['csv_row']['requested'] == 'mega'})
    return {'schema': 'megafmm.iter39.psplit-bn16-measurements.v1',
        'measurement_source_commit': identity['measurement_source_commit'],
        'frozen_sources': identity['frozen_sources'], 'frozen_binaries': identity['frozen_binaries'],
        'route_configurations': configurations, 'independent_compiled_gate': latest_gate,
        'compiled_gate_content_independently_reaudited_by_generator': False,
        'predeclared_plan': plan_binding,
        'measurement_scope': {'expected_row_count': 24, 'actual_row_count': len(rows),
            'complete_full_matrix': seen == expected,
            'expected_rows': [key_object(v) for v in sorted(expected)],
            'observed_rows': [key_object(v) for v in sorted(seen)],
            'missing_rows': [key_object(v) for v in sorted(missing)],
            'missing_csv_paths': missing_files, 'exclusions': exclusion_binding},
        'row_count': len(rows), 'timing_group_case_count': sum(len(z['raw_groups']) for z in rows),
        'fresh_gate_count': len(fresh), 'fresh_gate_pass': sum(c['pass'] == '1' for c in fresh),
        'fresh_gate_fail': sum(c['pass'] == '0' for c in fresh),
        'nonfinite': sum(int(z['csv_row']['nonfinite']) for z in rows), 'runs': runs, 'rows': rows,
        'same_binary_megakernel_comparison': comparisons39,
        'new_raw_bn16_vs_raw_bn32_comparison': comparisons40,
        'new_ss_bn16_vs_ss_bn32_comparison': comparisons36,
        'new_raw_bn16_vs_ss_bn16_comparison': comparisons43,
        'each_new_vs_retained_controls': each_new,
        'same_binary_own_latency_comparison': each_new,
        'own_latency_pair_contract': [{'candidate_variant': candidate, 'baseline_variant': baseline, 'axis': axis} for candidate, baseline, axis in pairs],
        'axis_comparisons': axes,
        'unavailable_comparisons': unavailable, 'observed_all_sw_route_winners': winners,
        'observed_native_sw_route_winners': native_winners,
        'dense_cv_at_least_2_percent': noisy,
        'actually_measured_native_variants': measured_native,
        'ordinary_two_current_pointer_encodes_inside_hostloop': False,
        'ordinary_encode_scope': 'Actually measured native variants ' + '/'.join(map(str, measured_native)) +
            ' rebuild transforms from current A/B inside one complete kernel, zero native matrixworkspace, no hostdescriptorencode. Lt1 is the separate complete multi-launch pack/leaf/combine control.',
        'graph_fixed_capture_bindings_fresh_contents': True,
        'tuning_protocol': 'Each row independently tunes ordinary cuBLASLt before capture; Dense and SW are timed in the same ordinary/Graph mode; no Graph-specific tuning mode.',
        'batchwall_protocol': 'Median five group batchhostwall values divided by100calls, including submission and batchsync; not independently synchronized singlecallwall.',
        'interpretation': 'Rank new42 rawBN16 and new43 SSBN16 by same-binary complete own SW latency. The predeclared unique five pairs preserve raw BN axis42vs40, SS BN axis43vs36, sameBN operand observation42vs43 and eachnew vs priorfastest39. BN/quota/minCTAs/tail differences do not certify isolated causes. Dense/SW is independently tuned same-row Dense eventmedian / complete SW eventmedian; denominator noise does not prove code benefit. Original gates/floors/Auto unchanged, FAIL remains FAIL, no precision or goal certification. Missing rows require exact compiled-blocked evidence and are not measurements.'}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN_DEFAULT)
    parser.add_argument('--identity', default=IDENTITY_DEFAULT)
    parser.add_argument('--exclusions', help='Identity and gate-SHA-bound new42/new43 complete blocked-four-key manifests; absent requires full24')
    parser.add_argument('--output', help='Owner-selected output; default stdout')
    args = parser.parse_args()
    encoded = json.dumps(generate(args.plan, args.identity, args.exclusions), indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(encoded)
    else:
        print(encoded, end='')
