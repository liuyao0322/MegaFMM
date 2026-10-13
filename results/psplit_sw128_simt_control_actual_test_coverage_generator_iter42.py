#!/usr/bin/env python3
"""CPU-only iteration42 audit; create actual coverage only after all logs close.

This script never builds, runs a GPU command, mutates Git, or changes formal
files. The frozen expected snapshot remains a prediction; actual counts come
from stdout and terminal sidecars. Run only after the owner gives terminal
notification for default, selected46, strict, disabled37 CLI, and all3 SAN.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / '.scratch/psplit_sw128_simt_control_test_expected_coverage_iter42.json'
EXPECTED_SHA = '42776ca9957fbdb197ca794dc90fd1c35f5bb88fe0fabb7fdb2b8bb8464a3be8'
OUTPUT = ROOT / '.scratch/psplit_sw128_simt_control_test_actual_coverage_iter42.json'
SOURCE_SHA = 'a6a9658cb4864ed72db1a519c363d9f08d6e237f3c4afb9c0838ea5580c0f6b3'
OLD_EXPECTED_SHA = '41b7e611253ace63e5a524d34913196dad6ddcdcee13eb9684db3c509045849d'
OLD_ACTUAL_SHA = 'b248599febac2455015685a8172f21c4b773caf452beec5ab5da30c638d75852'
FAMILIES = {46: 'WGMMA_PSPLIT_SW128_SIMT_CONTROL'}
NATIVES = {46: 'megakernel_wgmma_psplit_sw128_simt_control_64x32'}
PACKED = 'megakernel_warp_mma_packed'
DENSE = 'dense_cublaslt'
DTYPES = ('bf16', 'fp16')


class AuditError(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise AuditError(message)


def local(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display(path):
    return str(local(path).relative_to(ROOT))


def sha(path):
    with local(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def attrs(line):
    return dict(re.findall(r'([A-Za-z_0-9]+)=([^\s]+)', line))


def metric(value):
    number = float(value)
    return number if math.isfinite(number) else str(number)


def same_number(a, b):
    a, b = float(a), float(b)
    return a == b or (math.isnan(a) and math.isnan(b))


def formatted_close(a, b):
    a, b = float(a), float(b)
    return same_number(a, b) or math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-12)


def cases(lines, algorithm=None, variant=None):
    result = []
    for line in lines:
        if not line.startswith('CASE '):
            continue
        match = re.match(r'CASE (\S+) variant=(\d+) actual=(\S+) (bf16|fp16) (\S+) (\d+,\d+,\d+) ', line)
        check(match, 'Unrecognized CASE format: ' + line)
        if algorithm is not None and match[1] != algorithm:
            continue
        if variant is not None and int(match[2]) != variant:
            continue
        values = attrs(line)
        result.append({'algorithm': match[1], 'variant': int(match[2]),
                       'actual_route': match[3], 'dtype': match[4], 'pattern': match[5],
                       'shape': list(map(int, match[6].split(','))),
                       'metrics': {key: metric(values[key]) for key in
                                   ('maxabs', 'eF', 'scaled', 'normalized', 'nonfinite')}})
    return result


def markers(lines):
    return Counter(line for line in lines if line.startswith(('CASE ', 'PASS', 'BEGIN')))


def terminal(stem, allowed_exits, expected_command=None):
    path = ROOT / 'results' / (stem + '.txt')
    exit_path = path.with_suffix('.exit')
    check(path.is_file() and exit_path.is_file(), 'Terminal log/sidecar missing: ' + stem)
    exit_code = int(exit_path.read_text().strip())
    check(exit_code in allowed_exits, 'Unexpected exit for ' + stem + ': ' + str(exit_code))
    lines = path.read_text().splitlines()
    command_lines = [line for line in lines if line.startswith('COMMAND ')]
    check(len(command_lines) == 1, 'Need exactly one COMMAND line: ' + stem)
    command = json.loads(command_lines[0][8:])
    if expected_command is not None:
        check(command == expected_command, 'Command identity mismatch: ' + stem)
    record = {'path': display(path), 'sha256': sha(path), 'exit_path': display(exit_path),
              'exit_sha256': sha(exit_path), 'exit_code': exit_code, 'command': command,
              'CASE_lines': sum(line.startswith('CASE ') for line in lines),
              'PASS_lines': sum(line.startswith('PASS') for line in lines),
              'BEGIN_lines': sum(line.startswith('BEGIN') for line in lines)}
    return lines, record


def audit_identity(path, expected):
    path = local(path)
    identity = json.loads(path.read_text())
    bindings = {}
    for domain, count in (('frozen_sources', 7), ('frozen_binaries', 2)):
        check(len(identity[domain]) == count, 'Identity map count mismatch: ' + domain)
        bindings[domain] = {}
        for name, wanted in identity[domain].items():
            got = sha(name)
            check(got == wanted, 'Current frozen binding mismatch: ' + name)
            bindings[domain][name] = {'sha256': got, 'matches_frozen': True}
    check(identity['frozen_sources']['tests/test_correctness.cu'] == SOURCE_SHA == expected['source_sha256'],
          'Current test-source identity mismatch')
    commit = identity['measurement_source_commit']
    for name, wanted in identity['frozen_sources'].items():
        # git show is read-only. Check source commit blobs without changing a checkout.
        blob = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)
        check(hashlib.sha256(blob).hexdigest() == wanted, 'Source commit blob mismatch: ' + name)
    plan_descriptor = identity['predeclared_plan']
    check(sha(plan_descriptor['path']) == plan_descriptor['sha256'], 'Descriptor SHA mismatch: predeclared_plan')
    check(identity['predeclared_plan_sha256'] == plan_descriptor['sha256'], 'Top-level predeclared plan SHA differs')
    gate_field = 'compiled_gate' if 'compiled_gate' in identity else 'independent_compiled_gate'
    gate_descriptor = identity[gate_field]
    check(sha(gate_descriptor['path']) == gate_descriptor.get('sha256', gate_descriptor.get('SHA')),
          'Descriptor SHA mismatch: ' + gate_field)
    check(display(gate_descriptor['path']) == 'results/psplit_sw128_simt_control_static_review_initial_iter42.json',
          'Gate must be current iteration42 SW128 SIMT control artifact')
    gate = json.loads(local(gate_descriptor['path']).read_text())
    check(gate['status'] == gate_descriptor['status'] and gate['status'].startswith('PASS_'),
          'Current independent gate has not authorized registered SW128 SIMT control scope')
    # New measured schemas may omit linked_identity_path. In that case,
    # discover exactly one linked identity from the already frozen gate map.
    link_candidates = [name for name in gate['frozen_bindings'] if
                       name.startswith('results/psplit_sw128_simt_control_') and
                       'link' in name and name.endswith('_iter42.json')]
    if 'linked_identity_path' in identity:
        link_path = identity['linked_identity_path']
    elif gate.get('freshlink_path'):
        link_path = gate['freshlink_path']
    else:
        check(len(link_candidates) == 1, 'Need exactly one gate-bound SW128 SIMT control linked identity')
        link_path = link_candidates[0]
    check(gate['frozen_bindings'].get(display(link_path)) == sha(link_path),
          'Linked identity is not bound by the current independent compiled gate')
    link = json.loads(local(link_path).read_text())
    for name, wanted in identity['frozen_binaries'].items():
        check(link[name] == wanted, 'Linked identity mismatch: ' + name)
    result = {'path': display(path), 'sha256': sha(path), 'measurement_source_commit': commit,
              'base_checkpoint': identity['base_checkpoint'], 'verified_frozen_bindings': bindings,
              'source_count': len(bindings['frozen_sources']), 'binary_count': len(bindings['frozen_binaries']),
              'all7_source_commit_blobs_match_frozen_source_SHA': True,
              'predeclared_plan': identity['predeclared_plan'],
              'referenced_independent_compiled_gate': gate_descriptor,
              'compiled_gate_field_in_actual_measured_identity': gate_field,
              'compiled_gate_content_audit_by_this_test_agent': False,
              'linked_identity': {'path': display(link_path), 'sha256': sha(link_path)}}
    binary = {'path': 'build/test_correctness', 'sha256': identity['frozen_binaries']['build/test_correctness'],
              'source_sha256': SOURCE_SHA, 'matches_frozen_identity_and_linked_identity': True}
    return result, binary


def pass_rows(lines, variant, suffix):
    return [attrs(line) for line in lines if line.startswith('PASS_' + FAMILIES[variant] + suffix + ' ')]


def assertions(row, keys, context):
    check(all(row.get(key) == '1' for key in keys), 'Missing ' + context + ' assertion')


def route_audit(lines, variant, enabled, expected, must_complete):
    domain = expected['route_expected_domains'][str(variant)]
    prefix, native = FAMILIES[variant], NATIVES[variant]
    result = {'variant': variant, 'enabled': enabled[variant], 'complete': False}
    actual = cases(lines, 'mega', variant)
    begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + prefix + ' dtype=')]
    if not enabled[variant]:
        check(not actual and not any(line.startswith('BEGIN_' + prefix) for line in lines),
              'Blocked route executed native/fixture markers')
        result.update({'native_CASE_or_BEGIN': 0, 'basic_cases': None, 'rawU_hostepilogue': None,
                       'current_A_B_rebind': None, 'epilogue_rebind': None, 'Graph': None, 'policy': None, 'stride_metadata': None})
        return result
    basic = domain['new_expected_native_stage_case_domain']
    wanted = [(dtype, row) for dtype in DTYPES for row in basic['basic_cases_per_dtype']]
    check(len(actual) <= len(begins) <= len(wanted), 'Basic observed domain exceeds frozen source domain')
    route_names = {'native' + str(variant): native, 'packed_P7': PACKED, 'Dense': DENSE}
    for index, case in enumerate(actual):
        dtype, target = wanted[index]
        begin = begins[index]
        check(case['dtype'] == begin['dtype'] == dtype and case['shape'] == target['shape'] ==
              list(map(int, begin['shape'].split(','))), 'Basic ordered shape/dtype drift')
        pattern = 'distinct_eight_row_owners' if target['pattern'] == 'OwnerRows' else 'integers'
        check(case['pattern'] == pattern and case['actual_route'] == begin['expected'] == route_names[target['route']],
              'Basic original math pattern/actual route drift')
        check(float(begin['alpha']) == float(target['alpha']) and float(begin['beta']) == float(target['beta']) and
              bool(int(begin['alias'])) == target['alias'] and bool(int(begin['null_c0'])) == (float(target['beta']) == 0),
              'Basic alpha/beta/alias/nullC0 drift')
    exact = all(all(value == 0 for value in case['metrics'].values()) for case in actual)
    basic_complete = len(actual) == len(begins) == len(wanted) and exact
    result['basic_cases'] = {'CASE_lines': len(actual), 'by_actual_route': dict(Counter(row['actual_route'] for row in actual)),
                            'all_maxabs_eF_scaled_normalized_nonfinite_zero': exact,
                            'complete_ordered_domain': basic_complete, 'case_records': actual}
    ks = basic['parent_K']
    stage_cases = [row for row in actual if row['shape'][:2] == [128, 64] and row['shape'][2] in ks and row['actual_route'] == native]
    result['native_stage_domain'] = {'observed_parent_K': sorted({row['shape'][2] for row in stage_cases}),
                                    'stage_divisor': 128, 'stage_counts': sorted({row['shape'][2] // 128 for row in stage_cases}),
                                    'CASE_lines': len(stage_cases), 'stage8_K1024': any(row['shape'][2] == 1024 for row in stage_cases),
                                    'complete': len(stage_cases) == len(ks) * len(DTYPES)}
    bit_domain = domain['new_expected_rawU_hostepilogue']
    sameatom = True
    suffix = '_NATIVE_BITWISE' if sameatom else '_CROSSATOM_BITWISE'
    forbidden = '_CROSSATOM_BITWISE' if sameatom else '_NATIVE_BITWISE'
    check(not pass_rows(lines, variant, forbidden), 'Reference relationship mislabeled')
    reference_variant = 39
    reference_label = 'sameatom_N32_bk64_rsa39_raw_u+specified_host_epilogue'
    bits = pass_rows(lines, variant, suffix)
    bit_begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + prefix + suffix + ' ')]
    wanted_bits = {(dtype, tuple([*mn, k])) for dtype in DTYPES for mn in bit_domain['MN'] for k in ks}
    observed_bits = [(row['dtype'], tuple(map(int, row['shape'].split(',')))) for row in bits]
    check(len(bits) == len(set(observed_bits)) and set(observed_bits) <= wanted_bits and len(bits) <= len(bit_begins) <= len(wanted_bits),
          'Observed rawU domain duplicated/outside frozen scope')
    for row, begin in zip(bits, bit_begins):
        k = int(row['shape'].split(',')[2])
        check(int(row['variant']) == int(begin['variant']) == variant and
              int(row['reference_variant']) == int(begin['reference_variant']) == reference_variant and
              row['reference'] == reference_label, 'rawU reference/relationship drift')
        assertions(row, ('raw_u_bits', 'explicit_host_epilogue', 'padding_guard'), 'rawU/host oracle/guard')
        alpha, beta = (1, 0) if k == bit_domain['identity_K'] else (0.375, -0.625)
        check(float(row['alpha']) == float(begin['alpha']) == alpha and float(row['beta']) == float(begin['beta']) == beta and
              int(row['alias']) == int(begin['alias']) == (k == bit_domain['active_alias_K']) and
              int(row['null_c0']) == int(begin['null_c0']) == (k == bit_domain['identity_K']) and
              row['shape'] == begin['shape'] and row['dtype'] == begin['dtype'], 'rawU host epilogue parameters drift')
        shifted = k == bit_domain['aligned16_input_and_float4_output_K']
        check(int(begin['input_shift_bytes']) == (16 if shifted else 0) and
              int(begin['output_shift_bytes']) == (4 if shifted else 0), 'rawU address phases drift')
    identities = sum(float(row['alpha']) == 1 and float(row['beta']) == 0 for row in bits)
    mismatches = [attrs(line) for line in lines if line.startswith(prefix + '_BIT_MISMATCH ')]
    for row in mismatches:
        check(int(row['variant']) == variant and int(row['reference_variant']) == reference_variant and
              row['phase'] in ('raw_u', 'contract_epilogue'), 'Mismatch diagnostics reference drift')
    bits_complete = len(bits) == len(wanted_bits) and not mismatches
    result['rawU_hostepilogue'] = {'PASS_groups': len(bits), 'identity_groups': identities,
        'active_groups': len(bits) - identities,
        'kernel_runs_completed_by_PASS_bound_fixture_loops': identities * 2 + (len(bits) - identities) * 3,
        'reference_kernel_runs_completed_by_PASS_bound_fixture_loops': len(bits),
        'candidate_kernel_runs_completed_by_PASS_bound_fixture_loops': identities + (len(bits) - identities) * 2,
        'total_kernel_runs_include_reference': True,
        'BEGIN_groups': len(bit_begins), 'reference_variant': reference_variant,
        'reference_relationship': 'sameatom_N32_BK64',
        'reference_label': reference_label, 'PASS_suffix': suffix[1:], 'complete_unique_domain': bits_complete,
        'completed_PASS_groups_assert_rawU_bits_host_RNalpha_betaFMA_padding_guard': True,
        'mismatch_diagnostics': mismatches, 'unrelaxed_bit_comparison_failed': bool(mismatches),
        'kernel_runs_in_incomplete_BEGIN_not_claimed': True}
    rebind = pass_rows(lines, variant, '_INPUT_REBIND')
    check(len(rebind) <= 2 and len({row['dtype'] for row in rebind}) == len(rebind), 'A/B rebind duplicated')
    for row in rebind:
        check(int(row['variant']) == variant and row['dtype'] in DTYPES and
              (int(row['bindings']), int(row['native_bindings']), int(row['dense_bindings'])) == (10, 7, 3), 'A/B binding fields drift')
        assertions(row, ('independent_A_B_allocations', 'aligned16_phase', 'fresh_AB_each_binding',
                         'element_only_A_B_both_dense', 'aligned_recovery', 'trailing_guard'), 'A/B rebind')
    result['current_A_B_rebind'] = {'PASS_groups': len(rebind), 'bindings': sum(int(row['bindings']) for row in rebind),
        'native_bindings': sum(int(row['native_bindings']) for row in rebind),
        'Dense_bindings': sum(int(row['dense_bindings']) for row in rebind), 'complete': len(rebind) == 2}
    epi = pass_rows(lines, variant, '_EPILOGUE_REBIND')
    wanted_epi = {(dtype, alpha, beta, mode) for dtype in DTYPES for alpha, beta, modes in
                  ((0.5, -0.25, (0, 1, 2)), (1, 0, (0, 2))) for mode in modes}
    seen_epi = [(row['dtype'], float(row['alpha']), float(row['beta']), int(row['mode'])) for row in epi]
    check(len(epi) == len(set(seen_epi)) and set(seen_epi) <= wanted_epi, 'Epilogue rebind duplicated/outside domain')
    for row in epi:
        check(int(row['variant']) == variant and row['aligned_then_float_aligned_then_aligned'] == '1' and
              int(row['alias']) == (int(row['mode']) == 2), 'Epilogue recovery/alias assertion drift')
    result['epilogue_rebind'] = {'PASS_groups': len(epi), 'kernel_runs_completed_by_PASS_bound_fixture_loops': len(epi) * 3,
                                'complete_unique_domain': len(epi) == len(wanted_epi)}
    graphs = pass_rows(lines, variant, '_GRAPH')
    graph_begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + prefix + '_GRAPH ')]
    wanted_graph = {(dtype, tuple(shape)) for dtype in DTYPES for shape in domain['new_expected_graph']['shapes_per_dtype']}
    seen_graph = [(row['dtype'], tuple(map(int, row['shape'].split(',')))) for row in graphs]
    check(len(graphs) == len(set(seen_graph)) and set(seen_graph) <= wanted_graph and
          len(graphs) <= len(graph_begins) <= len(wanted_graph), 'Observed Graph domain drift')
    for row, begin in zip(graphs, graph_begins):
        m, n, k = map(int, row['shape'].split(','))
        route = native if m % 128 == n % 64 == k % 128 == 0 else PACKED
        identity, shifted = (m, n, k) == (128, 64, 128), (m, n, k) == (128, 256, 384)
        check(row['actual'] == begin['actual'] == route and int(row['variant']) == int(begin['variant']) == variant and
              row['dtype'] == begin['dtype'] and row['shape'] == begin['shape'] and row['kernel_nodes'] == '1' and
              row['refresh_ab'] == row['fixed_capture_AB_addresses'] == begin['refresh_ab'] == '1' and int(row['replays']) == 3 and
              int(begin['input_shift_bytes']) == (16 if shifted else 0) and int(begin['output_shift_bytes']) == (4 if shifted else 0) and
              int(begin['alias']) == int(row['alias']) == (not identity) and
              int(begin['null_c0']) == int(row['null_c0']) == identity, 'Graph pointer/route/node/replay domain drift')
    result['Graph'] = {'config_groups': len(graphs), 'actual_routes': dict(Counter(row['actual'] for row in graphs)),
        'fresh_AB_replays': sum(int(row['replays']) for row in graphs), 'complete_unique_domain': len(graphs) == len(wanted_graph),
        'one_kernel_node_each_completed_PASS': True, 'partial_K_P7': sorted({int(row['shape'].split(',')[2]) for row in graphs
        if row['actual'] == PACKED and row['shape'].startswith('128,64,')}),
        'immutable_C0_restore_asserted_by_frozen_fixture': True}
    policy = pass_rows(lines, variant, '_POLICY')
    wanted_policy = {'variant': str(variant), 'invalid47_cases': '20', 'validate_before_empty_scale': '1',
                     'empty_cases': '4', 'scale_cases': '8', 'auto_dense_dtypes': '2'}
    check(policy in ([], [wanted_policy]), 'BN16 policy fields drift')
    result['policy'] = {key: int(value) for key, value in policy[0].items()} if policy else None
    result['stride_metadata'] = audit_stride_metadata(lines, expected, must_complete)
    result['PASS_groups'] = sum(line.startswith('PASS_' + prefix) for line in lines)
    result['failure_lines'] = [line for line in lines if line.startswith('FAIL ')] if not must_complete else []
    result['complete'] = (basic_complete and bits_complete and len(rebind) == 2 and len(epi) == len(wanted_epi) and
                          len(graphs) == len(wanted_graph) and bool(policy) and result['stride_metadata']['complete'])
    if must_complete:
        check(result['complete'], 'Exit0 enabled BN16 route has incomplete/failing structural domain: ' + str(variant))
    return result


def audit_strict(lines, record, variants, expected):
    if not variants:
        check(record['exit_code'] == 1 and record['CASE_lines'] == record['PASS_lines'] == record['BEGIN_lines'] == 0 and
              any('requested SW128 SIMT control numerical scope has no enabled routes' in line for line in lines), 'Blocked strict CLI executed scope')
        record.update({'candidate_variants': [], 'candidate_gates': 0, 'candidate_PASS': 0, 'candidate_FAIL': 0,
                       'candidate_nonfinite': 0, 'Dense_reference_CASE': 0, 'status': 'BLOCKED_NOT_EXECUTED'})
        return record
    patterns = expected['route_expected_domains'][str(variants[0])]['new_expected_strict_original_gate']['patterns']
    trial_domain = [(dtype, pattern, trial) for dtype in DTYPES for pattern in patterns for trial in range(3)]
    gates = [attrs(line) for line in lines if line.startswith('NUMERICAL_GATE ')]
    candidate, dense = cases(lines, 'mega'), cases(lines, 'dense')
    distributions = [attrs(line) for line in lines if line.startswith('NUMERICAL_DISTRIBUTION ')]
    wanted = [(variant, *trial) for trial in trial_domain for variant in variants]
    check(len(gates) == len(candidate) == len(wanted) and len(dense) == len(trial_domain) and
          record['CASE_lines'] == len(candidate) + len(dense), 'Strict candidate/shared Dense domain incomplete')
    references = {}
    for key, row in zip(trial_domain, dense):
        check(row['dtype'] == key[0] and row['pattern'] == key[1] and row['shape'] == [256, 256, 256] and row['actual_route'] == DENSE,
              'Strict shared Dense reference order drift')
        references[key] = row
    rows, summary, routes, margins = [], {}, Counter(), []
    for identity, gate, case in zip(wanted, gates, candidate):
        variant, dtype, pattern, trial = identity
        reference = references[(dtype, pattern, trial)]
        check(gate['algorithm'] == 'mega' and int(gate['variant']) == case['variant'] == variant and
              gate['dtype'] == case['dtype'] == dtype and gate['pattern'] == case['pattern'] == pattern and
              int(gate['trial']) == trial and gate['shape'] == '256,256,256' and gate['k128_only'] == '0' and case['shape'] == [256, 256, 256],
              'Strict original shape/pattern/trial/variant scope drift')
        certified = pattern != 'fp16_upper_range'
        route = NATIVES[variant] if certified else DENSE
        check(bool(int(gate['certified_range'])) == certified and case['actual_route'] == route, 'Strict actual native/Dense fallback drift')
        floor = 0.001 if dtype == 'bf16' else 0.0001
        nonfinite = int(case['metrics']['nonfinite']) + int(reference['metrics']['nonfinite'])
        finite = nonfinite == 0
        dense_ef, solution_ef = float(gate['dense_ef']), float(gate['solution_ef'])
        limit = 3 * dense_ef + floor
        passed = finite and solution_ef <= limit
        check(float(gate['floor']) == floor and int(gate['nonfinite']) == nonfinite and bool(int(gate['finite'])) == finite and
              bool(int(gate['pass'])) == passed and formatted_close(gate['limit'], limit) and
              same_number(solution_ef, case['metrics']['eF']) and same_number(dense_ef, reference['metrics']['eF']),
              'Printed original numerical gate cannot be reproduced from CASE metrics')
        routes[route] += 1
        if math.isfinite(solution_ef) and math.isfinite(limit):
            margins.append(abs(solution_ef - limit))
        key = str(variant) + '/' + dtype + '/' + pattern
        group = summary.setdefault(key, {'variant': variant, 'dtype': dtype, 'pattern': pattern, 'gates': 0, 'PASS': 0, 'FAIL': 0,
                                         'nonfinite': 0, 'actual_route': route, 'solution_eF': [], 'dense_eF': []})
        group['gates'] += 1; group['PASS'] += int(passed); group['FAIL'] += int(not passed); group['nonfinite'] += nonfinite
        group['solution_eF'].append(metric(solution_ef)); group['dense_eF'].append(metric(dense_ef))
        rows.append({'variant': variant, 'dtype': dtype, 'pattern': pattern, 'trial': trial, 'actual_route': route,
                     'certified_range': certified, 'PASS': passed, 'finite': finite, 'nonfinite': nonfinite,
                     'solution_eF': metric(solution_ef), 'dense_eF': metric(dense_ef), 'floor': floor, 'recomputed_limit': metric(limit)})
    check(len(distributions) == len(summary), 'Strict distribution domain mismatch')
    for row in distributions:
        key = row['variant'] + '/' + row['dtype'] + '/' + row['pattern']
        check(key in summary, 'Strict distribution outside candidate scope')
        group = summary[key]
        check(int(row['trials']) == group['gates'] and int(row['failures']) == group['FAIL'] and int(row['nonfinite']) == group['nonfinite'],
              'Strict distribution disagrees with recomputed candidate gates')
    failures = sum(not row['PASS'] for row in rows)
    check(record['exit_code'] == (2 if failures else 0), 'Strict exit disagrees with actual candidate gate failures')
    by_variant = {}
    for variant in variants:
        subset = [row for row in rows if row['variant'] == variant]
        by_variant[str(variant)] = {'candidate_gates': len(subset), 'candidate_PASS': sum(row['PASS'] for row in subset),
            'candidate_FAIL': sum(not row['PASS'] for row in subset), 'candidate_nonfinite': sum(row['nonfinite'] for row in subset),
            'candidate_actual_routes': dict(Counter(row['actual_route'] for row in subset)),
            'native_gates': sum(row['actual_route'] == NATIVES[variant] for row in subset),
            'native_PASS': sum(row['actual_route'] == NATIVES[variant] and row['PASS'] for row in subset),
            'native_FAIL': sum(row['actual_route'] == NATIVES[variant] and not row['PASS'] for row in subset),
            'HighHalf_Dense_fallback_gates': sum(row['actual_route'] == DENSE for row in subset),
            'HighHalf_Dense_fallback_PASS': sum(row['actual_route'] == DENSE and row['PASS'] for row in subset)}
    record.update({'candidate_variants': variants, 'candidate_gates': len(rows), 'candidate_PASS': sum(row['PASS'] for row in rows),
        'candidate_FAIL': failures, 'candidate_nonfinite': sum(row['nonfinite'] for row in rows), 'candidate_actual_routes': dict(routes),
        'Dense_reference_CASE': len(dense), 'Dense_reference_shared_between_candidates': len(variants) > 1,
        'per_variant': by_variant, 'per_variant_dtype_pattern': summary, 'candidate_gate_records': rows,
        'unique_complete_dtype_pattern_trial_domain': True, 'all_gates_recomputed_from_printed_metrics': True,
        'min_absolute_gate_margin': min(margins) if margins else None,
        'gate': 'finite && solution_eF <= 3*dense_eF + floor', 'original_floor_BF16': 0.001, 'original_floor_FP16': 0.0001,
        'current_full_numerical_suite_actually_executed': False})
    return record


def audit_stride_metadata(lines, expected, must_complete):
    domain = expected['route_expected_domains']['46']['new_expected_stride_metadata']
    prefix = 'WGMMA_PSPLIT_SW128_SIMT_CONTROL_STRIDE_METADATA'
    begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + prefix + ' ')]
    actual = [attrs(line) for line in lines if line.startswith('META_' + prefix + ' ')]
    passes = [attrs(line) for line in lines if line.startswith('PASS_' + prefix + ' ')]
    wanted = [(dtype, scenario) for dtype in DTYPES for scenario in domain['scenarios_per_dtype']]
    check(len(actual) <= len(begins) <= len(wanted) and len(passes) <= 1, 'Observed stride metadata outside frozen domain')
    unavailable = {}
    for line in lines:
        if not line.startswith('SETUP_UNAVAILABLE_' + prefix + ' '):
            continue
        fields, message = line.split(' message=', 1)
        row = attrs(fields); key = (row['dtype'], int(row['scenario']))
        check(row['variant'] == '46' and key not in unavailable and
              (message.startswith('query Lt heuristics:') or message.startswith('create matrix layout:') or
               'cuBLASLt has no valid heuristic for this problem' in message), 'Unclassified/duplicated metadata setup error')
        unavailable[key] = message
    records = []
    for (dtype, target), begin, row in zip(wanted, begins, actual):
        scenario = target['scenario']; native = target['native_guard']; key = (dtype, scenario)
        check(row['dtype'] == begin['dtype'] == dtype and int(row['variant']) == int(begin['variant']) == 46 and
              int(row['scenario']) == int(begin['scenario']) == scenario and row['native_guard'] == begin['native_guard'] == str(int(native)) and
              row['expected'] == begin['expected'] == target['expected_route'] and
              int(row['lda']) == int(begin['lda']) == target['lda'] and int(row['ldb']) == int(begin['ldb']) == target['ldb'],
              'Stride metadata A-only helper predicate/shape/route intent drift')
        for field in ('safe_a_span_bytes', 'safe_b_span_bytes'):
            check(int(row[field]) == int(begin[field]) == target[field] and 0 <= int(row[field]) < 2**48,
                  'Stride metadata span is not the exact safe source span')
        check(row['metadata_only'] == begin['metadata_only'] == '1', 'Stride metadata-only marker missing')
        check(row['allocated_input_bytes'] == row['tensor_map_encodes'] == row['native_runs'] == begin['native_runs'] == '0',
              'Stride metadata executed allocated input, TMA encode, or a native kernel')
        supported = row['setup_supported'] == '1'
        check(row['setup_supported'] in ('0', '1') and (row['actual'] == target['expected_route'] if supported else
              row['actual'] == 'setup_unavailable') and (key not in unavailable if supported else key in unavailable),
              'Stride metadata setup_unavailable was falsely counted as successful Plan routing')
        records.append({'dtype': dtype, 'scenario': scenario, 'lda': target['lda'], 'ldb': target['ldb'], 'native_guard': native,
            'expected_route': target['expected_route'], 'actual_route': row['actual'], 'setup_supported': supported,
            'safe_a_span_bytes': target['safe_a_span_bytes'], 'safe_b_span_bytes': target['safe_b_span_bytes'],
            'setup_unavailable_message': unavailable.get(key), 'metadata_only': True, 'allocated_input_bytes': 0,
            'tensor_map_encodes': 0, 'native_runs': 0})
    check(set(unavailable) <= {(row['dtype'], row['scenario']) for row in records}, 'Setup error lacks completed metadata record')
    setup_success = sum(row['setup_supported'] for row in records)
    if passes:
        row = passes[0]
        check(len(actual) == len(begins) == len(wanted) and row['variant'] == '46' and
              (int(row['cases']), int(row['native_guard_cases']), int(row['packed_guard_cases']), int(row['dense_guard_cases'])) == (10, 4, 4, 2) and
              int(row['setup_success_cases']) == setup_success and int(row['setup_unavailable_cases']) == len(unavailable) and
              row['A_only_limit'] == str(2**39) and row['B_bound_not_applied'] == row['metadata_only'] ==
              row['native_helper_guards_all_checked'] == row['Plan_setup_claimed_only_if_supported'] == '1' and
              row['allocated_input_bytes'] == row['tensor_map_encodes'] == row['native_runs'] == '0', 'Stride PASS assertions/counts drift')
    complete = bool(passes) and len(records) == len(wanted)
    if must_complete:
        check(complete, 'Exit0 SW128 SIMT control lacks complete metadata boundary assertions')
    return {'PASS_groups': len(passes), 'cases': len(records), 'native_guard_cases': sum(row['native_guard'] for row in records),
        'packed_guard_cases': sum(row['expected_route'] == PACKED for row in records),
        'dense_guard_cases': sum(row['expected_route'] == DENSE for row in records),
        'setup_success_cases': setup_success, 'setup_unavailable_cases': len(unavailable),
        'B_bound_not_applied': complete, 'metadata_only': True, 'allocated_input_bytes': 0, 'tensor_map_encodes': 0,
        'native_runs': 0, 'successful_Plan_setup_actual_routes': dict(Counter(row['actual_route'] for row in records if row['setup_supported'])),
        'setup_unavailable_not_claimed_as_Plan_route_PASS': True, 'complete': complete, 'case_records': records}


def route_scope(*log_sets):
    states = []
    for lines in log_sets:
        for line in lines:
            if line.startswith('SW128_SIMT_CONTROL_ROUTE_SCOPE '):
                row = attrs(line)
                check(row['variant'] == '46' and row['enabled'] in ('0', '1') and row['dtypes'] == '2' and
                      row['metadata_only'] == '1' and row['native_runs'] == '0', 'Invalid SW128 SIMT control46 metadata scope')
                states.append(row['enabled'] == '1')
    check(states and len(set(states)) == 1, 'Closed metadata does not consistently establish46 enabled state')
    return {46: states[0]}


def policy_audit(lines, enabled, complete):
    blocked = [attrs(line) for line in lines if line.startswith('PASS_WGMMA_PSPLIT_SW128_SIMT_CONTROL_BLOCKED_POLICY ')]
    wanted = [] if enabled[46] else [{'variant': '46', 'forced_rejections': '8', 'validate_before_empty_scale': '1',
                                     'auto_dense_dtypes': '2', 'native_runs': '0'}]
    check(all(row in wanted for row in blocked) and len(blocked) <= 1, 'Blocked SW128 SIMT control policy drift')
    if complete:
        check(blocked == wanted, 'Completed structural scope lacks blocked46 policy')
    disabled = [attrs(line) for line in lines if line.startswith('PASS_WGMMA_PSPLIT_DISABLED_POLICY ')]
    expected37 = {'variant': '37', 'forced_rejections': '8', 'validate_before_empty_scale': '1',
                  'auto_dense_dtypes': '2', 'native_runs': '0', 'register_spill_gate_blocked': '1'}
    check(disabled in ([], [expected37]), 'Retained disabled37 policy drift')
    if complete:
        check(disabled == [expected37], 'Completed structural scope lacks disabled37 policy')
    check(not any(line.startswith('CASE mega variant=37 ') or (line.startswith('BEGIN') and 'variant=37 ' in line)
                  for line in lines), 'Disabled37 execution marker found')
    return {'blocked_new_variant_policy': [{key: int(value) for key, value in row.items()} for row in blocked],
            'retained_disabled37_policy': {key: int(value) for key, value in disabled[0].items()} if disabled else None,
            'native37_CASE_or_BEGIN': 0}


def structural(lines, record, enabled, expected, selected=False):
    if selected and not enabled[46]:
        check(record['exit_code'] == 1 and record['CASE_lines'] == record['BEGIN_lines'] == record['PASS_lines'] == 0 and
              any('FAIL Mega46 is disabled:' in line for line in lines), 'Blocked46 selected CLI launched fixtures')
        record.update({'variant': 46, 'enabled': False, 'native_runs': 0,
                       'rejected_before_any_CASE_BEGIN': True, 'structural_status': 'BLOCKED_NOT_EXECUTED'})
        return record
    complete = record['exit_code'] == 0
    route = route_audit(lines, 46, enabled, expected, complete and enabled[46])
    record.update(policy_audit(lines, enabled, complete))
    record['new_routes'] = {'46': route}
    if selected:
        record.update(route)
    graphs = [attrs(line) for line in lines if re.match(r'^PASS_[^\s]*GRAPH\b', line)]
    record['all_Graph'] = {'config_groups': len(graphs), 'fresh_AB_replays': sum(int(row['replays']) for row in graphs)}
    record['structural_status'] = 'PASS' if complete else 'FAIL_INCOMPLETE_OR_BIT_MISMATCH'
    record['failure_lines'] = [line for line in lines if line.startswith('FAIL ')]
    if not complete:
        check(record['failure_lines'], 'Nonzero structural exit lacks explicit failure evidence')
    return record


def audit_sanitizers(enabled, selected_lines):
    if not enabled[46]:
        return {'status': 'no_enabled_routes_no_SAN_executed', 'completed_runs': 0,
                'candidate_variants': [], 'all_SAN_clean': None, 'reports': []}
    reports = []
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        lines, record = terminal('sanitizer_psplit_sw128_simt_control46_' + tool + '_iter42', {0, 1, 99})
        command, text = record['command'], '\n'.join(lines)
        extra = ['--leak-check', 'full'] if tool == 'memcheck' else ['--racecheck-report', 'all'] if tool == 'racecheck' else []
        check(command[:2] == ['bash', 'scripts/on_gpu.sh'] and command[2].endswith('/compute-sanitizer') and
              command[3:] == ['--tool', tool, '--error-exitcode', '99'] + extra +
              ['./build/test_correctness', '--smoke-candidates', '--mega-only', '--mega-variant', '46'], 'Sanitizer executable/scope mismatch')
        version = re.findall(r'^Version (\S+) \(build (\d+)\) \(public-release\)$', text, re.M)
        check(len(version) == 1, 'Sanitizer version missing/duplicated')
        functional_equal = markers(lines) == markers(selected_lines)
        if record['exit_code'] == 0:
            check(functional_equal, 'Exit0 SAN scope differs from closed selected46')
        errors = list(map(int, re.findall(r'ERROR SUMMARY:\s*(\d+) errors', text)))
        leaks = re.findall(r'LEAK SUMMARY:\s*(\d+) bytes leaked in (\d+) allocations', text)
        races = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
        if tool == 'racecheck':
            check(len(races) == 1 and not errors and not leaks, 'Racecheck terminal summary mismatch')
            hazards, error_count, warnings = map(int, races[0]); kind = 'RACECHECK SUMMARY'
        else:
            check(len(errors) == 1 and not races, 'Sanitizer terminal ERROR summary mismatch')
            error_count, hazards, warnings, kind = errors[0], None, None, 'ERROR SUMMARY'
        if tool == 'memcheck':
            check(len(leaks) == 1, 'Memcheck full LEAK summary missing')
            leaked_bytes, allocations = map(int, leaks[0])
        else:
            check(not leaks, 'Non-memcheck leak evidence must remain unmeasured')
            leaked_bytes = allocations = None
        record.update({'variant': 46, 'tool': tool, 'sanitizer_version': version[0][0], 'sanitizer_build': version[0][1],
            'CASE_PASS_BEGIN_multisets_equal_closed_selected_log': functional_equal, 'error_summary_kind': kind,
            'errors': error_count, 'race_hazards': hazards, 'race_warnings': warnings,
            'leak_check_measured': tool == 'memcheck', 'leaked_bytes': leaked_bytes, 'leaked_allocations': allocations,
            'rawU_PASS_groups': len(pass_rows(lines, 46, '_NATIVE_BITWISE')), 'rawU_reference_relationship': 'sameatom_N32_BK64',
            'Graph_configs': len(pass_rows(lines, 46, '_GRAPH')), 'native37_CASE_or_BEGIN': 0})
        reports.append(record)
    check(len({(row['sanitizer_version'], row['sanitizer_build']) for row in reports}) == 1, 'SAN versions differ')
    clean = all(row['exit_code'] == 0 and row['errors'] == 0 and row['race_hazards'] in (None, 0) and
                row['race_warnings'] in (None, 0) and row['leaked_bytes'] in (None, 0) and row['leaked_allocations'] in (None, 0)
                for row in reports)
    return {'status': 'closed_all3_variant46_reports_verified', 'completed_runs': len(reports), 'candidate_variants': [46],
            'all_SAN_clean': clean, 'race_sync_leaks_unmeasured': True, 'reports': reports}


def legacy_audit(lines):
    prior_path = 'results/correctness_psplit_selective_tma_a_iter41.txt'
    old_lines = local(prior_path).read_text().splitlines()
    identity = lambda row: (row['algorithm'], row['variant'], row['actual_route'], row['dtype'], row['pattern'], tuple(row['shape']))
    current = [row for row in cases(lines) if not (row['algorithm'] == 'mega' and row['variant'] == 46)]
    previous = cases(old_lines)
    retained = Counter(map(identity, current)) == Counter(map(identity, previous))
    current_pass = Counter(line for line in lines if line.startswith('PASS') and not line.startswith('PASS_' + FAMILIES[46]))
    previous_pass = Counter(line.replace('invalid46_cases', 'invalid47_cases') for line in old_lines if line.startswith('PASS'))
    pass_retained = current_pass == previous_pass
    check(retained and pass_retained, 'Old0..45 domains/markers changed beyond invalid47 metadata')
    return {'prior_default_path': prior_path, 'prior_default_sha256': sha(prior_path),
            'old_CASE_identity_multiset_retained': retained, 'old_PASS_multiset_retained_after_invalid46_to47_metadata': pass_retained,
            'old_CASE_lines': len(current), 'old_PASS_lines': sum(current_pass.values()),
            'prior_source_sha256': 'da1f6e02386f3f6366637e31e30d22c4ccb459ed98db222307b7b83f22444473',
            'prior_actual_snapshot_sha256': OLD_ACTUAL_SHA, 'old_numeric_metric_bitwise_identity_claim': False}


def run(args):
    check(not OUTPUT.exists(), 'Actual output already exists; never overwrite frozen evidence')
    check(sha(EXPECTED) == EXPECTED_SHA, 'Expected snapshot changed')
    old_expected = '.scratch/psplit_selective_tmaA_test_expected_coverage_iter41.json'
    old_actual = '.scratch/psplit_selective_tmaA_test_actual_coverage_iter41.json'
    check(sha(old_expected) == OLD_EXPECTED_SHA and sha(old_actual) == OLD_ACTUAL_SHA, 'Historical snapshots changed')
    if args.formal_expected:
        check(sha(args.formal_expected) == EXPECTED_SHA, 'Formal expected copy differs')
    expected = json.loads(EXPECTED.read_text())
    result = json.loads(EXPECTED.read_text())
    identity, binary = audit_identity(args.identity, expected)
    commands = expected['target_commands']
    default_lines, default = terminal('correctness_psplit_sw128_simt_control_iter42', {0, 1}, commands['default'])
    selected_lines, selected = terminal('correctness_psplit_sw128_simt_control46_targeted_iter42', {0, 1}, commands['selected46'])
    enabled = route_scope(default_lines, selected_lines)
    actual_identity = json.loads(local(args.identity).read_text())
    descriptor = actual_identity.get('compiled_gate', actual_identity.get('independent_compiled_gate'))
    gate = json.loads(local(descriptor['path']).read_text())
    allowed = gate.get('allowed_native_launch_variants', actual_identity.get('allowed_native_launch_variants'))
    check(allowed is not None and sorted(map(int, allowed)) == ([46] if enabled[46] else []),
          'Closed Plan metadata enable set differs from independent compiled gate allowed_native_launch_variants')
    identity['allowed_native_launch_variants'] = sorted(map(int, allowed))
    identity['Plan_metadata_enable_set_matches_independent_compiled_gate'] = True
    default = structural(default_lines, default, enabled, expected)
    selected = structural(selected_lines, selected, enabled, expected, True)
    legacy = legacy_audit(default_lines)
    strict_lines, strict = terminal('strict_psplit_sw128_simt_control_iter42', {0, 1, 2}, commands['strict_new46_only'])
    strict = audit_strict(strict_lines, strict, [46] if enabled[46] else [], expected)
    cli_lines, cli = terminal('disabled_psplit_sw128_simt_control37_cli_iter42', {1}, commands['disabled37_cli'])
    check(cli['CASE_lines'] == cli['PASS_lines'] == cli['BEGIN_lines'] == 0 and
          any('FAIL Mega37 is disabled:' in line for line in cli_lines), 'Disabled37 CLI did not reject before fixtures')
    cli.update({'native_runs': 0, 'rejected_before_any_BEGIN_CASE': True, 'explicit_disabled37_reason_matched': True})
    sanitizers = audit_sanitizers(enabled, selected_lines)
    result['actual_validation'] = {'default': default, 'selected46': selected, 'strict_sw128_simt_control_enabled_only': strict,
        'disabled37_cli': cli, 'sanitizers': sanitizers, 'measured_source_identity': identity, 'compiled_test_binary_identity': binary}
    result['legacy_default_domain_reaudit'] = legacy
    result['compiled_enabled_variant_set_actual'] = [46] if enabled[46] else []
    structural_pass = default['exit_code'] == 0 and (not enabled[46] or selected['exit_code'] == 0)
    result['status'] = 'actual_validation_closed_structural_' + ('PASS' if structural_pass else 'FAIL') + '_SAN_' + (
        'UNEXECUTED' if not enabled[46] else 'PASS' if sanitizers['all_SAN_clean'] else 'FAIL') + '_original_strict_' + str(strict['candidate_FAIL']) + 'FAIL'
    result['actual_scope_summary'] = {'enabled_variants': result['compiled_enabled_variant_set_actual'],
        'default_CASE': default['CASE_lines'], 'default_PASS': default['PASS_lines'], 'default_exit': default['exit_code'],
        'selected46_CASE': selected['CASE_lines'], 'selected46_PASS': selected['PASS_lines'], 'selected46_exit': selected['exit_code'],
        'default_all_Graph_configs': default['all_Graph']['config_groups'], 'default_all_Graph_fresh_AB_replays': default['all_Graph']['fresh_AB_replays'],
        'strict_candidate_gates': strict['candidate_gates'], 'strict_PASS': strict['candidate_PASS'], 'strict_FAIL': strict['candidate_FAIL'],
        'strict_nonfinite': strict['candidate_nonfinite'], 'strict_exit': strict['exit_code'], 'sanitizer_completed_runs': sanitizers['completed_runs'],
        'full_numerical_code_domain': 1830 if enabled[46] else 1800, 'full_numerical_code_domain_actually_executed': False,
        'original_full_numerical_certification_claimed': False, 'performance_claim_by_this_artifact': False}
    result['authorized_write_scope'] = [display(OUTPUT)]
    result['pre_execution_expected_snapshot'] = {'path': display(EXPECTED), 'sha256': EXPECTED_SHA, 'original_bytes_preserved': True}
    if args.formal_expected:
        result['pre_execution_expected_snapshot']['formal_copy'] = {'path': display(args.formal_expected), 'sha256': EXPECTED_SHA}
    result['audit_generator_identity'] = {'path': display(__file__), 'sha256': sha(__file__), 'CPU_only': True,
        'no_GPU_build_formal_write_or_git_mutation': True, 'prepared_after_source_freeze_before_actual_validation': True}
    result['limitations'] = [
        'All actual counts derive from closed stdout/sidecars. Expected counts are separate source-domain predictions.',
        '46 uses N32 BK64 RS-A39 rawU plus independent volatile float RNalpha and std::fma betaC0 epilogue; bits are never relaxed.',
        'rawU run totals include39 reference launches; candidate and reference runs are separately derived from completed PASS fixture loops.',
        'Only new46 original numerical gates were executed; full1830 and historical complete code domains remain unexecuted.',
        'Huge-stride tests are host metadata-only:zero input allocation, TMA encode and native runs; setup_unavailable preserves actual Dense-backend messages and is not successful Plan routing.',
        'Historical source41 executions retain separate identities. Structural/SAN success does not certify original FP64 numerical gates.',
        'Only full memcheck measures leaks; racecheck and synccheck leak fields remain null.',
        'Only46 is newly registered;47 invalid and37 disabled have no native/performance execution claims. CurrentA host encoding is retained although the SIMT kernel does not use the tensor map.']
    evidence = [default, selected, strict, cli, *sanitizers['reports']]
    for record in evidence:
        check(sha(record['path']) == record['sha256'] and sha(record['exit_path']) == record['exit_sha256'], 'Terminal evidence changed during CPU audit')
    check(sha(EXPECTED) == EXPECTED_SHA and sha(args.identity) == identity['sha256'] and sha(__file__) == result['audit_generator_identity']['sha256'],
          'Frozen generator/expected/identity changed during CPU audit')
    check(sha(old_expected) == OLD_EXPECTED_SHA and sha(old_actual) == OLD_ACTUAL_SHA, 'Historical snapshots changed during CPU audit')
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output_path': display(OUTPUT), 'output_sha256': sha(OUTPUT), 'status': result['status'],
                      'actual_scope_summary': result['actual_scope_summary']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', default='results/psplit_sw128_simt_control_measured_identity_iter42.json')
    parser.add_argument('--formal-expected', help='Optional frozen formal expected copy, verified without mutation')
    args = parser.parse_args()
    try:
        run(args)
    except (AuditError, FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, 'CPU AUDIT NOT FINALIZED: ' + str(error) + '\n')
