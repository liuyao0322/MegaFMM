#!/usr/bin/env python3
"""CPU-only iteration36 audit; create actual coverage only after all logs close.

This script never builds, runs a GPU command, mutates Git, or changes formal
files. The frozen expected snapshot remains a prediction; actual counts come
from stdout and terminal sidecars. Run only after the owner gives terminal
notification for default, selected39, strict, disabled37 CLI, and all3 SAN.
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
EXPECTED = ROOT / '.scratch/psplit_rsa_expected_test_coverage_iter36.json'
EXPECTED_SHA = '7c9992d6689dae2ca7d35f0e3254541cd3c1c08036847b3c2a2407fc7d1bb21a'
OUTPUT = ROOT / '.scratch/psplit_rsa_actual_test_coverage_iter36.json'
SOURCE_SHA = '4158a44c10d06bf9eb10278c5b4efbdae22ce010298a8db7c8dd3daef1c4eb69'
OLD_EXPECTED_SHA = '88e07de12ecb06c73818c833e7291e776cd27b9ba193a28f8a26edcdd65cabb0'
OLD_ACTUAL_SHA = '1e2918af20abd537386c8d83b36363355c91777979f41fd47e128f326cee557d'
PREFIX = 'WGMMA_PSPLIT_RSA'
NATIVE = 'megakernel_wgmma_psplit_rsa_64x32'
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


def pass_attrs(lines, suffix):
    return [attrs(line) for line in lines if line.startswith('PASS_' + PREFIX + suffix + ' ')]


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
    for field in ('predeclared_plan', 'independent_compiled_gate'):
        descriptor = identity[field]
        check(sha(descriptor['path']) == descriptor['sha256'], 'Descriptor SHA mismatch: ' + field)
    gate_descriptor = identity['independent_compiled_gate']
    check(display(gate_descriptor['path']) == 'results/psplit_rsa_static_review_initial_iter36.json',
          'Gate must be current iteration36 RSA artifact')
    gate = json.loads(local(gate_descriptor['path']).read_text())
    check(gate['status'] == gate_descriptor['status'] and gate['status'].startswith('PASS_COMPILED_CAN_LAUNCH_39'),
          'Current independent gate has not authorized native39')
    link_path = identity['linked_identity_path']
    link = json.loads(local(link_path).read_text())
    for name, wanted in identity['frozen_binaries'].items():
        check(link[name] == wanted, 'Linked identity mismatch: ' + name)
    result = {'path': display(path), 'sha256': sha(path), 'measurement_source_commit': commit,
              'base_checkpoint': identity['base_checkpoint'], 'verified_frozen_bindings': bindings,
              'source_count': len(bindings['frozen_sources']), 'binary_count': len(bindings['frozen_binaries']),
              'all7_source_commit_blobs_match_frozen_source_SHA': True,
              'predeclared_plan': identity['predeclared_plan'],
              'referenced_independent_compiled_gate': gate_descriptor,
              'compiled_gate_content_audit_by_this_test_agent': False,
              'linked_identity': {'path': display(link_path), 'sha256': sha(link_path)}}
    binary = {'path': 'build/test_correctness', 'sha256': identity['frozen_binaries']['build/test_correctness'],
              'source_sha256': SOURCE_SHA, 'matches_frozen_identity_and_linked_identity': True}
    return result, binary


def audit_structural(lines, record, expected):
    new_cases = cases(lines, 'mega', 39)
    begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + PREFIX + ' dtype=')]
    domain = expected['new_expected_native_stage_case_domain']
    wanted = [(dtype, case) for dtype in DTYPES for case in domain['basic_cases_per_dtype']]
    check(len(new_cases) == len(begins) == len(wanted), 'New39 basic domain incomplete/duplicated')
    routes = {'native39': NATIVE, 'packed_P7': PACKED, 'Dense': DENSE}
    for (dtype, case), begin, actual in zip(wanted, begins, new_cases):
        check(actual['dtype'] == begin['dtype'] == dtype and actual['shape'] == case['shape'] ==
              list(map(int, begin['shape'].split(','))), 'Basic shape/dtype mismatch')
        pattern = 'distinct_eight_row_owners' if case['pattern'] == 'OwnerRows' else 'integers'
        check(actual['pattern'] == pattern, 'Basic input pattern mismatch')
        check(actual['actual_route'] == begin['expected'] == routes[case['route']], 'Basic actual route mismatch')
        check(float(begin['alpha']) == float(case['alpha']) and float(begin['beta']) == float(case['beta']) and
              bool(int(begin['alias'])) == case['alias'] and int(begin['null_c0']) == (float(case['beta']) == 0),
              'Basic epilogue/alias/nullC0 mismatch')
        check(all(value == 0 for value in actual['metrics'].values()), 'Basic integer output is not exact/finite')
    record['new39_basic_cases'] = {'CASE_lines': len(new_cases),
        'by_actual_route': dict(Counter(case['actual_route'] for case in new_cases)),
        'all_maxabs_eF_scaled_normalized_nonfinite_zero': True,
        'ordered_shape_pattern_params_route_match_frozen_source_domain': True, 'case_records': new_cases}
    ks = domain['parent_K']
    stage_cases = [case for case in new_cases if case['shape'][:2] == [128, 64] and
                   case['shape'][2] in ks and case['actual_route'] == NATIVE]
    check(len(stage_cases) == len(ks) * len(DTYPES) and {(case['dtype'], case['shape'][2]) for case in stage_cases} ==
          {(dtype, k) for dtype in DTYPES for k in ks}, 'Native stage domain incomplete')
    actual_ks = sorted({case['shape'][2] for case in stage_cases})
    record['native_stage_domain'] = {'dtypes': list(DTYPES), 'parent_K': actual_ks,
        'stage_counts': [k // 128 for k in actual_ks], 'stage8_K1024': 1024 in actual_ks,
        'exact_one_stage_case_per_dtype_K': True, 'long_K_bounded_integer_pattern_checked': True}
    bits = pass_attrs(lines, '_NATIVE_BITWISE')
    bit_domain = expected['new_expected_sameatom_bitwise']
    wanted_bits = {(dtype, tuple([*mn, k])) for dtype in DTYPES for mn in bit_domain['MN'] for k in ks}
    check(len(bits) == len(wanted_bits) and {(row['dtype'], tuple(map(int, row['shape'].split(',')))) for row in bits} ==
          wanted_bits, 'rawU/host epilogue domain incomplete/duplicated')
    bit_begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + PREFIX + '_NATIVE_BITWISE ')]
    check(len(bit_begins) == len(bits), 'rawU BEGIN/PASS mismatch')
    for row, begin in zip(bits, bit_begins):
        k = int(row['shape'].split(',')[2])
        check(row['variant'] == begin['variant'] == '39' and row['reference_variant'] == begin['reference_variant'] == '38',
              'Sameatom reference must be explicit BK64 SS38')
        check(row['reference'] == 'sameatom_product_split_bk64_ss_raw_u+specified_host_epilogue' and
              all(row[key] == '1' for key in ('raw_u_bits', 'explicit_host_epilogue', 'padding_guard')),
              'Missing rawU/independent host oracle/guard assertions')
        check(int(row['alias']) == (k == bit_domain['active_alias_K']) and
              int(row['null_c0']) == (k == bit_domain['identity_K']), 'rawU alias/nullC0 domain mismatch')
        wanted_alpha, wanted_beta = (1, 0) if k == bit_domain['identity_K'] else (0.375, -0.625)
        check(float(row['alpha']) == float(begin['alpha']) == wanted_alpha and
              float(row['beta']) == float(begin['beta']) == wanted_beta and
              row['alias'] == begin['alias'] and row['null_c0'] == begin['null_c0'],
              'rawU BEGIN/PASS epilogue fields disagree with the frozen domain')
        check(row['shape'] == begin['shape'] and row['dtype'] == begin['dtype'] and
              int(begin['input_shift_bytes']) == (16 if k == bit_domain['aligned16_input_and_float4_output_K'] else 0) and
              int(begin['output_shift_bytes']) == (4 if k == bit_domain['aligned16_input_and_float4_output_K'] else 0),
              'rawU current input/output address phase mismatch')
    identity = sum(float(row['alpha']) == 1 and float(row['beta']) == 0 for row in bits)
    record['sameatom_rawU_hostepilogue'] = {'PASS_groups': len(bits), 'identity_groups': identity,
        'active_groups': len(bits) - identity, 'kernel_runs_completed_by_bound_fixture_loop': identity * 2 + (len(bits) - identity) * 3,
        'reference_variant': 38, 'reference_storage': 'BK64 SS', 'raw_U_bits_all_checked': True,
        'independent_host_RNalpha_betaFMA_all_checked': True, 'padding_guard_all_checked': True,
        'unique_complete_domain': True, 'native_parent_K': actual_ks, 'MN': bit_domain['MN']}
    rebind = pass_attrs(lines, '_INPUT_REBIND')
    check(len(rebind) == len(DTYPES) and {row['dtype'] for row in rebind} == set(DTYPES), 'A/B rebind domain mismatch')
    for row in rebind:
        check(row['variant'] == '39' and all(row[key] == '1' for key in
              ('independent_A_B_allocations', 'aligned16_phase', 'fresh_AB_each_binding',
               'element_only_A_B_both_dense', 'aligned_recovery', 'trailing_guard')), 'A/B rebind assertions missing')
        check(int(row['bindings']) == int(row['native_bindings']) + int(row['dense_bindings']) == 10 and
              int(row['native_bindings']) == 7 and int(row['dense_bindings']) == 3, 'A/B fixture binding count mismatch')
    record['current_A_B_rebind'] = {'PASS_groups': len(rebind), 'bindings': sum(int(row['bindings']) for row in rebind),
        'native_bindings': sum(int(row['native_bindings']) for row in rebind),
        'Dense_bindings': sum(int(row['dense_bindings']) for row in rebind),
        'fresh_current_pointer_and_aligned_recovery_assertions_completed': True}
    epi = pass_attrs(lines, '_EPILOGUE_REBIND')
    wanted_epi = {(dtype, alpha, beta, mode) for dtype in DTYPES for alpha, beta, modes in
                  ((0.5, -0.25, (0, 1, 2)), (1, 0, (0, 2))) for mode in modes}
    check(len(epi) == len(wanted_epi) and {(row['dtype'], float(row['alpha']), float(row['beta']), int(row['mode']))
          for row in epi} == wanted_epi, 'Scalar D/C0/alias epilogue domain incomplete/duplicated')
    check(all(row['variant'] == '39' and row['aligned_then_float_aligned_then_aligned'] == '1' for row in epi),
          'Epilogue recovery assertion missing')
    check(all(int(row['alias']) == (int(row['mode']) == 2) for row in epi), 'Epilogue alias marker mismatch')
    record['epilogue_rebind'] = {'PASS_groups': len(epi), 'kernel_runs_completed_by_bound_fixture_loop': len(epi) * 3,
        'unique_complete_identity_active_D_C0_alias_domain': True}
    graphs = pass_attrs(lines, '_GRAPH')
    wanted_graph = {(dtype, tuple(shape)) for dtype in DTYPES for shape in expected['new_expected_graph']['shapes_per_dtype']}
    check(len(graphs) == len(wanted_graph) and {(row['dtype'], tuple(map(int, row['shape'].split(',')))) for row in graphs} ==
          wanted_graph, 'Graph domain incomplete/duplicated')
    graph_begins = [attrs(line) for line in lines if line.startswith('BEGIN_' + PREFIX + '_GRAPH ')]
    check(len(graph_begins) == len(graphs), 'Graph BEGIN/PASS domain mismatch')
    for row, begin in zip(graphs, graph_begins):
        m, n, k = map(int, row['shape'].split(','))
        wanted_route = NATIVE if m % 128 == n % 64 == k % 128 == 0 else PACKED
        check(row['actual'] == wanted_route and row['variant'] == '39' and row['kernel_nodes'] == '1' and
              row['refresh_ab'] == row['fixed_capture_AB_addresses'] == '1' and int(row['replays']) == 3,
              'Graph actual route/node/fresh input/replay mismatch')
        identity_graph = (m, n, k) == (128, 64, 128)
        shifted_graph = (m, n, k) == (128, 256, 384)
        check(begin['dtype'] == row['dtype'] and begin['shape'] == row['shape'] and begin['actual'] == row['actual'] and
              begin['variant'] == '39' and begin['refresh_ab'] == '1' and
              int(begin['input_shift_bytes']) == (16 if shifted_graph else 0) and
              int(begin['output_shift_bytes']) == (4 if shifted_graph else 0) and
              int(begin['alias']) == int(row['alias']) == (not identity_graph) and
              int(begin['null_c0']) == int(row['null_c0']) == identity_graph,
              'Graph current-pointer/alias/nullC0 stdout fields mismatch')
    all_graphs = [attrs(line) for line in lines if re.match(r'^PASS_[^\s]*GRAPH\b', line)]
    record['Graph'] = {'new39_config_groups': len(graphs), 'new39_actual_routes': dict(Counter(row['actual'] for row in graphs)),
        'new39_fresh_AB_replays': sum(int(row['replays']) for row in graphs), 'one_kernel_node_each': True,
        'unique_complete_domain': True, 'native_parent_K': actual_ks,
        'partial_K_P7': sorted({int(row['shape'].split(',')[2]) for row in graphs if row['actual'] == PACKED and row['shape'].startswith('128,64,')}),
        'immutable_C0_restore_verified_in_bound_fixture_source': True,
        'all_default_or_selected_config_groups': len(all_graphs),
        'all_default_or_selected_fresh_replays': sum(int(row['replays']) for row in all_graphs)}
    policy = pass_attrs(lines, '_POLICY')
    check(policy == [{'variant': '39', 'invalid40_cases': '20', 'validate_before_empty_scale': '1',
                     'empty_cases': '4', 'scale_cases': '8', 'auto_dense_dtypes': '2'}], 'New39 policy domain mismatch')
    record['new39_policy'] = {key: int(value) for key, value in policy[0].items()}
    disabled = [attrs(line) for line in lines if line.startswith('PASS_WGMMA_PSPLIT_DISABLED_POLICY ')]
    check(disabled == [{'variant': '37', 'forced_rejections': '8', 'validate_before_empty_scale': '1',
                       'auto_dense_dtypes': '2', 'native_runs': '0', 'register_spill_gate_blocked': '1'}],
          'Retained disabled37 policy domain mismatch')
    record['retained_disabled37_policy'] = {key: int(value) for key, value in disabled[0].items()}
    check(not any(line.startswith('CASE mega variant=37 ') or (line.startswith('BEGIN') and 'variant=37 ' in line)
                  for line in lines), 'Disabled37 native execution marker found')
    record['native37_CASE_or_BEGIN'] = 0
    record['new39_PASS_groups_excluding_disabled37_and_banner'] = sum(line.startswith('PASS_' + PREFIX) for line in lines)
    return record


def audit_strict(lines, record, expected):
    gates = [attrs(line) for line in lines if line.startswith('NUMERICAL_GATE ')]
    candidate, dense = cases(lines, 'mega'), cases(lines, 'dense')
    distributions = [attrs(line) for line in lines if line.startswith('NUMERICAL_DISTRIBUTION ')]
    domain = expected['new_expected_strict_psplit_rsa_only']
    wanted = {(dtype, pattern, trial) for dtype in DTYPES for pattern in domain['patterns']
              for trial in range(domain['fresh_trials_per_pattern'])}
    check(len(gates) == len(candidate) == len(dense) == len(wanted) and record['CASE_lines'] == len(gates) + len(dense),
          'Strict candidate/reference domain incomplete/duplicated')
    check({(row['dtype'], row['pattern'], int(row['trial'])) for row in gates} == wanted, 'Strict dtype/pattern/trial mismatch')
    rows, summary, routes, margins = [], {}, Counter(), []
    for gate, case, reference in zip(gates, candidate, dense):
        check(gate['algorithm'] == 'mega' and gate['variant'] == '39' and gate['shape'] == '256,256,256' and
              gate['k128_only'] == '0' and case['variant'] == 39, 'Strict candidate scope drift')
        check(case['dtype'] == reference['dtype'] == gate['dtype'] and case['pattern'] == reference['pattern'] == gate['pattern'] and
              case['shape'] == reference['shape'] == [256, 256, 256], 'Strict CASE/gate pair mismatch')
        certified = gate['pattern'] != 'fp16_upper_range'
        route = NATIVE if certified else DENSE
        check(int(gate['certified_range']) == certified and case['actual_route'] == route and reference['actual_route'] == DENSE,
              'Strict actual native/HighHalf fallback route mismatch')
        routes[route] += 1
        floor = 0.001 if gate['dtype'] == 'bf16' else 0.0001
        check(float(gate['floor']) == floor, 'Original strict floor changed')
        nonfinite = int(case['metrics']['nonfinite']) + int(reference['metrics']['nonfinite'])
        finite = nonfinite == 0
        check(int(gate['nonfinite']) == nonfinite and int(gate['finite']) == finite, 'Strict finite/NF fields disagree')
        dense_ef, solution_ef = float(gate['dense_ef']), float(gate['solution_ef'])
        limit = 3 * dense_ef + floor
        passed = finite and solution_ef <= limit
        check(int(gate['pass']) == passed and formatted_close(gate['limit'], limit), 'Printed strict gate cannot be reproduced')
        check(same_number(solution_ef, case['metrics']['eF']) and same_number(dense_ef, reference['metrics']['eF']),
              'Strict CASE/gate eF mismatch')
        if math.isfinite(solution_ef) and math.isfinite(limit):
            margins.append(abs(solution_ef - limit))
        key = gate['dtype'] + '/' + gate['pattern']
        group = summary.setdefault(key, {'gates': 0, 'PASS': 0, 'FAIL': 0, 'nonfinite': 0,
                                         'actual_route': route, 'solution_eF': [], 'dense_eF': []})
        group['gates'] += 1; group['PASS'] += int(passed); group['FAIL'] += int(not passed); group['nonfinite'] += nonfinite
        group['solution_eF'].append(metric(solution_ef)); group['dense_eF'].append(metric(dense_ef))
        rows.append({'dtype': gate['dtype'], 'pattern': gate['pattern'], 'trial': int(gate['trial']),
                     'actual_route': route, 'certified_range': certified, 'PASS': passed, 'finite': finite,
                     'nonfinite': nonfinite, 'dense_eF': metric(dense_ef), 'solution_eF': metric(solution_ef),
                     'floor': floor, 'recomputed_limit': metric(limit)})
    check(len(distributions) == len(summary), 'Strict distribution scope mismatch')
    check({row['dtype'] + '/' + row['pattern'] for row in distributions} == set(summary),
          'Strict distribution domain incomplete/duplicated')
    for row in distributions:
        group = summary[row['dtype'] + '/' + row['pattern']]
        check(row['variant'] == '39' and int(row['trials']) == group['gates'] and int(row['failures']) == group['FAIL'] and
              int(row['nonfinite']) == group['nonfinite'], 'Strict distributions disagree with gates')
    failures = sum(not row['PASS'] for row in rows)
    check(record['exit_code'] == (2 if failures else 0), 'Strict terminal exit disagrees with actual gates')
    record.update({'candidate_gates': len(rows), 'candidate_PASS': sum(row['PASS'] for row in rows),
        'candidate_FAIL': failures, 'candidate_nonfinite': sum(row['nonfinite'] for row in rows),
        'candidate_actual_routes': dict(routes), 'native_gates': sum(row['actual_route'] == NATIVE for row in rows),
        'native_PASS': sum(row['actual_route'] == NATIVE and row['PASS'] for row in rows),
        'native_FAIL': sum(row['actual_route'] == NATIVE and not row['PASS'] for row in rows),
        'HighHalf_Dense_fallback_gates': sum(row['actual_route'] == DENSE for row in rows),
        'HighHalf_Dense_fallback_PASS': sum(row['actual_route'] == DENSE and row['PASS'] for row in rows),
        'Dense_reference_CASE': len(dense), 'unique_complete_dtype_pattern_trial_domain': True,
        'all_gates_recomputed_from_printed_metrics': True, 'all_distributions_match_recomputed_gates': True,
        'min_absolute_gate_margin': min(margins) if margins else None,
        'gate': 'finite && solution_eF <= 3*dense_eF + floor', 'original_floor_BF16': 0.001,
        'original_floor_FP16': 0.0001, 'per_dtype_pattern': summary, 'candidate_gate_records': rows,
        'source36_full1620_actually_executed': False})
    return record


def audit_sanitizers(selected_lines):
    reports = []
    selected_markers = markers(selected_lines)
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        lines, record = terminal('sanitizer_psplit_rsa39_' + tool + '_iter36', {0, 99})
        command, text = record['command'], '\n'.join(lines)
        check(command[:2] == ['bash', 'scripts/on_gpu.sh'] and command[2].endswith('/compute-sanitizer'),
              'Sanitizer executable identity mismatch')
        extra = ['--leak-check', 'full'] if tool == 'memcheck' else ['--racecheck-report', 'all'] if tool == 'racecheck' else []
        check(command[3:] == ['--tool', tool, '--error-exitcode', '99'] + extra +
              ['./build/test_correctness', '--smoke-candidates', '--mega-only', '--mega-variant', '39'],
              'Sanitizer actual candidate/flags mismatch')
        version = re.findall(r'^Version (\S+) \(build (\d+)\) \(public-release\)$', text, re.M)
        check(len(version) == 1, 'Sanitizer version header missing/duplicated')
        check(markers(lines) == selected_markers, 'Sanitizer functional scope differs from selected39: ' + tool)
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
            check(len(leaks) == 1, 'Full memcheck LEAK summary missing/duplicated')
            leaked_bytes, leaked_allocs = map(int, leaks[0])
        else:
            check(not leaks, 'Unexpected leak evidence from a non-leak-check tool')
            leaked_bytes = leaked_allocs = None
        record.update({'variant': 39, 'tool': tool, 'sanitizer_version': version[0][0], 'sanitizer_build': version[0][1],
            'complete_CASE_PASS_BEGIN_multisets_equal_closed_selected39_log': True, 'error_summary_kind': kind,
            'errors': error_count, 'race_hazards': hazards, 'race_warnings': warnings,
            'leak_check_measured': tool == 'memcheck', 'leaked_bytes': leaked_bytes, 'leaked_allocations': leaked_allocs,
            'new39_rawU_groups': len(pass_attrs(lines, '_NATIVE_BITWISE')),
            'new39_Graph_configs': len(pass_attrs(lines, '_GRAPH')), 'native37_CASE_or_BEGIN': 0})
        reports.append(record)
    check(len({(row['sanitizer_version'], row['sanitizer_build']) for row in reports}) == 1,
          'Sanitizer versions differ within the serial validation set')
    clean = all(row['exit_code'] == 0 and row['errors'] == 0 and row['race_hazards'] in (None, 0) and
                row['race_warnings'] in (None, 0) and row['leaked_bytes'] in (None, 0) and
                row['leaked_allocations'] in (None, 0) for row in reports)
    return {'status': 'closed_all3_variant39_reports_verified', 'completed_runs': len(reports),
            'candidate_variants': [39], 'no37_sanitizer_execution': True,
            'all_exit0_errors0': all(row['exit_code'] == 0 and row['errors'] == 0 for row in reports),
            'memcheck_full_zero_leaks': next(row['leaked_bytes'] == row['leaked_allocations'] == 0 for row in reports if row['tool'] == 'memcheck'),
            'race_hazards_warnings_zero': next(row['race_hazards'] == row['race_warnings'] == 0 for row in reports if row['tool'] == 'racecheck'),
            'race_sync_leaks_unmeasured': True, 'all_SAN_clean': clean, 'reports': reports}


def run(args):
    check(not OUTPUT.exists(), 'Output already exists; do not overwrite a frozen actual artifact')
    check(sha(EXPECTED) == EXPECTED_SHA, 'Frozen expected snapshot changed')
    check(sha('.scratch/psplit_bk64_test_coverage_iter35.json') == OLD_EXPECTED_SHA and
          sha('.scratch/psplit_bk64_actual_test_coverage_iter35.json') == OLD_ACTUAL_SHA, 'Historical snapshots changed')
    if args.formal_expected:
        check(sha(args.formal_expected) == EXPECTED_SHA, 'Formal expected copy changed')
    expected = json.loads(EXPECTED.read_text())
    result = json.loads(EXPECTED.read_text())
    result['authorized_write_scope'] = ['.scratch/psplit_rsa_actual_test_coverage_iter36.json']
    result['pre_execution_expected_snapshot'] = {'path': display(EXPECTED), 'sha256': EXPECTED_SHA, 'original_bytes_preserved': True}
    if args.formal_expected:
        result['pre_execution_expected_snapshot']['formal_copy'] = {'path': display(args.formal_expected), 'sha256': EXPECTED_SHA}
    result['audit_generator_identity'] = {'path': display(__file__), 'sha256': sha(__file__),
                                         'CPU_only': True, 'no_GPU_build_formal_write_or_git_mutation': True}
    identity, binary = audit_identity(args.identity, expected)
    commands = expected['target_commands']
    default_lines, default = terminal('correctness_psplit_rsa_iter36', {0}, commands['default'])
    selected_lines, selected = terminal('correctness_psplit_rsa39_targeted_iter36', {0}, commands['selected39'])
    default = audit_structural(default_lines, default, expected)
    selected = audit_structural(selected_lines, selected, expected)
    check(selected['CASE_lines'] == selected['new39_basic_cases']['CASE_lines'], 'Other selected39 CASE domain found')
    banners = sum(line.startswith('PASS selected WGMMA product-split RS-A integer DAG,') for line in selected_lines)
    check(banners == 1 and selected['PASS_lines'] == selected['new39_PASS_groups_excluding_disabled37_and_banner'] +
          len([line for line in selected_lines if line.startswith('PASS_WGMMA_PSPLIT_DISABLED_POLICY ')]) + banners,
          'Selected39 PASS scope contains unaccounted markers')
    old_actual = json.loads(local('.scratch/psplit_bk64_actual_test_coverage_iter35.json').read_text())
    old_default = old_actual['actual_validation']['default']
    old_log = local(old_default['path'])
    check(sha(old_log) == old_default['sha256'] and sha(old_default['exit_path']) == old_default['exit_sha256'] and
          int(local(old_default['exit_path']).read_text().strip()) == old_default['exit_code'] == 0,
          'Historical source35 default evidence differs from its frozen actual identity')
    old_lines = old_log.read_text().splitlines()
    old_cases = Counter(line.split(' maxabs=')[0] for line in old_lines if line.startswith('CASE '))
    retained_cases = Counter(line.split(' maxabs=')[0] for line in default_lines if line.startswith('CASE ') and
                             not line.startswith('CASE mega variant=39 '))
    old_pass = Counter(line.replace('invalid39_cases', 'invalid40_cases') for line in old_lines if line.startswith('PASS'))
    retained_pass = Counter(line for line in default_lines if line.startswith('PASS') and not line.startswith('PASS_' + PREFIX))
    check(old_cases == retained_cases and old_pass == retained_pass, 'Old0..38 actual fixture domain drift')
    legacy = {'CASE_identity_prefix_multisets_equal': True, 'legacy_CASE': sum(old_cases.values()),
              'legacy_PASS': sum(old_pass.values()),
              'legacy_PASS_multisets_equal_after_only_invalid39_to40_metadata_normalization': True,
              'prior_log_path': display(old_log), 'prior_log_sha256': sha(old_log),
              'prior_test_source_sha256': old_actual['source_sha256'],
              'prior_actual_snapshot_sha256': OLD_ACTUAL_SHA,
              'old_numeric_metric_bitwise_identity_claim': False}
    strict_lines, strict = terminal('strict_psplit_rsa_iter36', {0, 2}, commands['strict_new39_only'])
    strict = audit_strict(strict_lines, strict, expected)
    cli_lines, cli = terminal('disabled_psplit_rsa37_cli_iter36', {1}, commands['disabled37_cli'])
    check(cli['CASE_lines'] == cli['PASS_lines'] == cli['BEGIN_lines'] == 0 and
          'FAIL Mega37 is disabled: asymmetric consumer register/spill gate failed before launch' in cli_lines,
          'Disabled37 CLI must reject before all CASE/BEGIN markers')
    cli.update({'native_runs': 0, 'explicit_disabled37_reason_matched': True, 'rejected_before_any_BEGIN_or_CASE': True})
    sanitizers = audit_sanitizers(selected_lines)
    result['actual_validation'] = {'default': default, 'selected39': selected, 'disabled37_cli': cli,
                                  'strict_psplit_rsa_only': strict, 'sanitizers': sanitizers,
                                  'measured_source_identity': identity, 'compiled_test_binary_identity': binary,
                                  'legacy_default_domain_reaudit': legacy}
    strict_outcome = 'original_strict' + str(strict['candidate_FAIL']) + 'FAIL' if strict['candidate_FAIL'] else 'original_strict_PASS'
    result['status'] = 'actual_validation_closed_structural_' + ('SAN_PASS_' if sanitizers['all_SAN_clean'] else 'SAN_FAIL_') + strict_outcome
    result['actual_scope_summary'] = {
        'default_CASE': default['CASE_lines'], 'default_PASS': default['PASS_lines'],
        'selected39_CASE': selected['CASE_lines'], 'selected39_PASS': selected['PASS_lines'],
        'new39_basic_actual_routes': selected['new39_basic_cases']['by_actual_route'],
        'rawU_groups': selected['sameatom_rawU_hostepilogue']['PASS_groups'],
        'rawU_bound_loop_kernel_runs': selected['sameatom_rawU_hostepilogue']['kernel_runs_completed_by_bound_fixture_loop'],
        'A_B_bindings': selected['current_A_B_rebind']['bindings'],
        'epilogue_PASS_groups': selected['epilogue_rebind']['PASS_groups'],
        'epilogue_bound_loop_runs': selected['epilogue_rebind']['kernel_runs_completed_by_bound_fixture_loop'],
        'Graph_configs': selected['Graph']['new39_config_groups'], 'Graph_actual_routes': selected['Graph']['new39_actual_routes'],
        'Graph_fresh_AB_replays': selected['Graph']['new39_fresh_AB_replays'],
        'default_all_Graph_configs': default['Graph']['all_default_or_selected_config_groups'],
        'default_all_Graph_fresh_AB_replays': default['Graph']['all_default_or_selected_fresh_replays'],
        'strict_candidate_gates': strict['candidate_gates'], 'strict_PASS': strict['candidate_PASS'],
        'strict_FAIL': strict['candidate_FAIL'], 'strict_nonfinite': strict['candidate_nonfinite'], 'strict_exit': strict['exit_code'],
        'sanitizer_variant39_runs': sanitizers['completed_runs'], 'sanitizer_all_clean': sanitizers['all_SAN_clean'],
        'disabled37_CLI_exit': cli['exit_code'], 'disabled37_CLI_native_CASE': cli['CASE_lines'],
        'source36_full1620_actually_executed': False, 'original_new39_only_gate_all_pass': strict['candidate_FAIL'] == 0,
        'original_full_numerical_suite_certification_claimed': False,
        'performance_claim_by_this_test_artifact': False}
    result['limitations'] = [
        'Frozen EXPECTED7c999 is preserved as a separate pre-execution prediction; all current actual counts come from terminal stdout and sidecars.',
        'Only original new39-only candidate gates were executed; full1620 and historical full1590 remain unexecuted complete code domains.',
        'Historical source29 full1290, source30-to33 new60 each, source34/35 new30 each retain separate execution identities.',
        'Structural integer/rawU/hostepilogue/Graph and sanitizer outcomes do not replace the separately recorded original FP64 numerical gate.',
        'Only full memcheck measured leaks; racecheck and synccheck leak metrics remain null.',
        'Compiled gate SHA/status and final source/bin maps are verified for identity; compiled correctness and performance are not re-audited by this CPU test artifact.',
        'Mega37 forced policy/CLI stay blocked with no native CASE/BEGIN or timing claim.']
    # Recheck all identity/log bindings immediately before the only output write.
    for domain in ('frozen_sources', 'frozen_binaries'):
        for name, binding in identity['verified_frozen_bindings'][domain].items():
            check(sha(name) == binding['sha256'], 'Binding changed during CPU audit: ' + name)
    for record in (default, selected, strict, cli, *sanitizers['reports']):
        check(sha(record['path']) == record['sha256'] and sha(record['exit_path']) == record['exit_sha256'],
              'Terminal evidence changed during CPU audit')
    check(sha(EXPECTED) == EXPECTED_SHA and sha(args.identity) == identity['sha256'], 'Frozen audit inputs changed')
    check(sha('.scratch/psplit_bk64_test_coverage_iter35.json') == OLD_EXPECTED_SHA and
          sha('.scratch/psplit_bk64_actual_test_coverage_iter35.json') == OLD_ACTUAL_SHA,
          'Historical snapshots changed during CPU audit')
    if args.formal_expected:
        check(sha(args.formal_expected) == EXPECTED_SHA, 'Formal expected copy changed during CPU audit')
    check(sha(__file__) == result['audit_generator_identity']['sha256'], 'Generator changed during CPU audit')
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output_path': display(OUTPUT), 'output_sha256': sha(OUTPUT),
                      'status': result['status'], 'actual_scope_summary': result['actual_scope_summary']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', default='results/psplit_rsa_measured_identity_iter36.json')
    parser.add_argument('--formal-expected', help='Optional frozen formal copy of EXPECTED7c999 to verify')
    args = parser.parse_args()
    try:
        run(args)
    except (AuditError, FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, 'CPU AUDIT NOT FINALIZED: ' + str(error) + '\n')
