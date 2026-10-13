#!/usr/bin/env python3
"""Owner-run dynamic iteration37 evidence profile generator.

Reads actual archives only. The reviewer prepares this scratch script; the
formal owner selects --output and executes it. It does not substitute for the
independent audit or for the compiled legality review. Missing validation stays
null/pending and every historical compile attempt is retained in raw_sha256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE = 'ecc92db8601a9a98d8338b115ee7dd0f7ae679d5'
PRIOR = 'results/small_psplit_rsa_profile_iter36.json'
MAPS = ('source_hashes', 'binaries', 'old_cuda_sha256', 'retained_old_evidence_sha256', 'prior_csv_sha256', 'raw_sha256')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_json(path):
    return json.loads(subprocess.check_output(['git', 'show', f'{BASE}:{path}']))


def fields(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))


def stdout_object(path):
    text = Path(path).read_text()
    start = text.find('{')
    require(start >= 0, 'Missing actual model stdout JSON: ' + str(path))
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def actual_log(record):
    if record is None:
        return None
    path = Path(record['path'])
    text = path.read_text()
    code = int(path.with_suffix('.exit').read_text())
    return {'path': str(path), 'exit': code, 'CASE': len(re.findall(r'^CASE ', text, re.M)),
            'PASS': len(re.findall(r'^PASS', text, re.M))}


def bound_compiled_gate(plan, identity, metrics):
    gate = identity.get('independent_compiled_gate')
    require(isinstance(gate, dict) and all(gate.get(key) for key in ('path', 'sha256', 'status')),
            'Require the actual current iteration37 independent compiled gate')
    require(metrics.get('independent_compiled_gate') == gate,
            'Metrics and identity must bind the same latest original compiled gate')
    require(sha(gate['path']) == gate['sha256'], 'Original compiled gate onsite SHA mismatch')
    require(not any(word in str(gate['status']).upper() for word in ('PENDING', 'UNKNOWN')),
            'The original compiled gate status must not be pending')
    original = json.loads(Path(gate['path']).read_text())
    require('iter37' in original.get('schema', '') and 'compiled' in original.get('schema', ''),
            'Require an iteration37 original compiled review schema')
    require(original.get('status') == gate['status'], 'Original gate and identity status mismatch')
    require(original.get('compiled_gate_inherited_from_iteration36') is False,
            'Original gate must explicitly declare no inherited iteration36 certification')
    bindings = original.get('frozen_bindings', {})
    for path, wanted in {**plan['frozen_sources'], **plan['frozen_binaries']}.items():
        require(bindings.get(path) == wanted, 'Original compiled gate source/binary binding mismatch: ' + path)
    return dict(gate)



def generate(plan_path, identity_path, measurements_path, coverage_path, compiled_path=None,
             environment_path=None, report_path=None):
    plan = json.loads(Path(plan_path).read_text())
    identity = json.loads(Path(identity_path).read_text())
    metrics = json.loads(Path(measurements_path).read_text())
    coverage = json.loads(Path(coverage_path).read_text())
    require(plan['frozen_sources'] == identity['frozen_sources'] and plan['frozen_binaries'] == identity['frozen_binaries'],
            'Plan/identity frozen maps differ')
    require(len(plan['frozen_sources']) == 7 and len(plan['frozen_binaries']) == 2, 'Require actual seven-source/two-binary freeze')
    expected_sources = {'solution/megakernel_wgmma_psplit_rsa_bk32.cu', 'solution/plan.cpp',
                        'include/megafmm/kernels.h', 'include/megafmm/gemm.h', 'CMakeLists.txt',
                        'bench/benchmark.cu', 'tests/test_correctness.cu'}
    require(set(plan['frozen_sources']) == expected_sources, 'Require all six common files and the distinct new RS-A TU')
    require(tuple(plan[key] for key in ('warmup', 'trials', 'groups', 'correct_trials', 'rotate_inputs',
                                       'heuristic_candidates', 'workspace_limit')) == (20, 100, 5, 3, 1, 32, 8589934592),
            'Mandatory full measurement protocol changed')
    require(metrics['measurement_scope']['expected_row_count'] == 16, 'Four routes/two dtypes/two timing modes target16')
    require(identity['measurement_source_commit'] == metrics['measurement_source_commit'], 'Metrics measured-source mismatch')
    require(metrics['frozen_binaries'] == plan['frozen_binaries'], 'Metrics frozen-binary mismatch')
    require(metrics.get('frozen_sources') == plan['frozen_sources'], 'Metrics frozen-source mismatch')
    require(metrics['predeclared_plan'] == {'path': str(plan_path), 'sha256': sha(plan_path)}, 'Metrics plan identity mismatch')
    require(metrics.get('route_configurations', {}) == plan.get('route_configurations', {}), 'Metrics final route configurations mismatch')
    require(isinstance(plan.get('route_configurations', {}).get('40', {}).get('enabled'), bool),
            'Final40 enabled must be explicitly frozen from fresh compiled gate and public API scope')
    require(coverage['source_sha256'] == plan['frozen_sources']['tests/test_correctness.cu'], 'Coverage final source identity mismatch')
    declared_plan_hashes = []
    if identity.get('predeclared_plan_sha256'):
        declared_plan_hashes.append(identity['predeclared_plan_sha256'])
    for name in ('plan', 'predeclared_plan'):
        declaration = identity.get(name)
        if isinstance(declaration, dict) and declaration.get('sha256'):
            require(declaration.get('path', str(plan_path)) == str(plan_path), 'Identity plan path mismatch')
            declared_plan_hashes.append(declaration['sha256'])
    require(declared_plan_hashes and all(value == sha(plan_path) for value in declared_plan_hashes),
            'Every declared plan SHA must bind the actual measured plan')
    gate = bound_compiled_gate(plan, identity, metrics)
    for path, wanted in {**plan['frozen_sources'], **plan['frozen_binaries']}.items():
        require(sha(path) == wanted, 'Current frozen source/binary mismatch: ' + path)
    prior = base_json(PRIOR)
    prior_metrics = base_json('results/small_psplit_rsa_measurements_iter36.json')
    old_cuda = {}
    for path in sorted(Path('solution').glob('*.cu')):
        if path.name == 'megakernel_wgmma_psplit_rsa_bk32.cu':
            continue
        data = subprocess.check_output(['git', 'show', f'{BASE}:{path}'])
        require(path.read_bytes() == data, 'Old CUDA source drift: ' + str(path))
        old_cuda[str(path)] = sha(path)
    require(len(old_cuda) == 33, 'Require all33 old CUDA sources')
    retained = {**prior['retained_old_evidence_sha256'], **prior['raw_sha256']}
    extra = [PRIOR, 'results/small_psplit_rsa_measurements_iter36.json', 'results/1024_MEGAKERNEL_PSPLIT_RSA_REPORT.md',
             'results/strict_psplit_rsa_iter36.txt', 'results/strict_psplit_rsa_iter36.exit',
             'results/psplit_rsa_test_coverage_iter36.json', 'results/psplit_rsa_static_review_final_iter36.json',
             'results/psplit_rsa_static_review_final_iter36.md', 'results/psplit_rsa_data_audit_iter36.json']
    for path in extra:
        retained[path] = hashlib.sha256(subprocess.check_output(['git', 'show', f'{BASE}:{path}'])).hexdigest()
    tracked = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], text=True).splitlines())
    retained_tracked, retained_onsite = [], []
    for path, wanted in retained.items():
        require(sha(path) == wanted, 'Retained old evidence drift: ' + path)
        if path in tracked:
            require(hashlib.sha256(subprocess.check_output(['git', 'show', f'{BASE}:{path}'])).hexdigest() == wanted,
                    'Retained base Git object mismatch: ' + path)
            retained_tracked.append(path)
        else:
            retained_onsite.append(path)
    require(len(retained_onsite) == 4, 'Only original four Nsys artifacts remain onsite-only')
    prior_csv = dict(prior['prior_csv_sha256'])
    prior_csv.update({entry['path']: entry['sha256'] for entry in prior['quick_measurements']})
    prior_csv.update({entry['path']: entry['sha256'] for entry in prior_metrics['runs']})
    for path, wanted in prior_csv.items():
        require(sha(path) == wanted, 'Prior CSV drift: ' + path)
    actual = coverage['actual_validation']
    for label, record in actual.items():
        if not record or not isinstance(record, dict):
            continue
        if 'source_sha256' in record:
            require(record['source_sha256'] == plan['frozen_sources']['tests/test_correctness.cu'], 'Actual coverage source mismatch: ' + label)
        if 'binary_sha256_at_read' in record:
            require(record['binary_sha256_at_read'] == plan['frozen_binaries']['build/test_correctness'], 'Actual coverage binary mismatch: ' + label)
    validation = {'default': actual_log(actual.get('default')),
                  'selected40': actual_log(actual.get('selected40')),
                  'disabled37_cli': actual_log(actual.get('disabled37_cli')),
                  'disabled40_cli': actual_log(actual.get('disabled40_cli')),
                  'disabled_policy_coverage_path': str(coverage_path),
                  'test_coverage_audit_path': str(coverage_path), 'strict': None,
                  'execution_scope': 'Only archived current-source runs are actual; blocked or absent scopes stay null. Full1650 is code scope, not an executed rerun. Historical source34 new36-only30, source35 new38-only30 and source36 new39-only30 remain separately bound.'}
    validation['CPU_checks'] = {}
    for label in ('DAG', 'stage'):
        path = Path(f'results/psplit_rsa_bk32_cpu_{label.lower()}_iter37.txt')
        if not path.exists():
            validation['CPU_checks'][label + '_scope'] = 'Actual current archived run absent; no PASS or exit inferred.'
            continue
        counts = re.findall(r'Ran (\d+) tests', path.read_text())
        require(len(counts) == 1, 'Actual CPU unittest count absent/ambiguous: ' + str(path))
        validation['CPU_checks'].update({label + '_path': str(path), label + '_tests': int(counts[0]),
                                         label + '_exit': int(path.with_suffix('.exit').read_text())})
    validation['CPU_state_model'] = None
    path = Path('results/psplit_rsa_bk32_cpu_state_model_iter37.txt')
    if path.exists():
        value = stdout_object(path)
        validation['CPU_state_model'] = {'path': str(path), 'exit': int(path.with_suffix('.exit').read_text()),
             'exit_recorded': True, 'exit_path': str(path.with_suffix('.exit')), 'status': value.get('status'),
             'stdout_summary': value, 'scope': 'Actual original RS-A source/abstract-model stdout and independently archived process exit. The owner archives a runner that removes only the old model tail-write section and protects original RS files with an audit hook; original main output writes must not run. Abstract/source scope only, not fresh compiled register allocation, GPU memory-order or liveness certification.',
             'not_hardware_liveness_certificate': True}
    default = validation['default']
    if default:
        lines = Path(default['path']).read_text().splitlines()
        new_graphs = [fields(line) for line in lines if line.startswith('PASS') and 'GRAPH' in line.split()[0] and fields(line).get('variant') == '40' and 'replays=' in line]
        all_graphs = [fields(line) for line in lines if line.startswith('PASS') and 'GRAPH' in line.split()[0] and 'replays=' in line]
        bindings = [fields(line) for line in lines if line.startswith('PASS') and 'INPUT_REBIND' in line.split()[0] and fields(line).get('variant') == '40']
        validation.update({'new_graph_configurations': len(new_graphs),
                           'new_graph_fresh_replays': sum(int(value['replays']) for value in new_graphs),
                           'total_graph_configurations': len(all_graphs),
                           'total_graph_fresh_replays': sum(int(value['replays']) for value in all_graphs),
                           'sameatom_rawU_host_epilogue_groups': sum(line.startswith('PASS') and 'NATIVE_BITWISE' in line.split()[0] and fields(line).get('variant') == '40' for line in lines),
                           'current_pointer_binding_checks': sum(int(value['bindings']) for value in bindings),
                           'sameatom_scope': 'Only actually executed enabled40 sameatom checks use the reference variant declared by current frozen coverage, then independent host RNalpha/betaFMA; disabled37 is never a native execution claim. No N16 oracle or separately exported sevenP claim.'})
    else:
        validation.update({field: None for field in ('new_graph_configurations', 'new_graph_fresh_replays', 'total_graph_configurations',
                              'total_graph_fresh_replays', 'sameatom_rawU_host_epilogue_groups', 'current_pointer_binding_checks')})
    strict_groups = []
    strict_record = actual.get('strict_psplit_rsa_bk32_only', actual.get('strict_rsa_bk32_only'))
    if strict_record:
        path = Path(strict_record['path'])
        lines = path.read_text().splitlines()
        gates, routes = [], []
        for at, line in enumerate(lines):
            if line.startswith('NUMERICAL_GATE '):
                gates.append(fields(line))
                routes.append(re.search(r'actual=(\S+)', lines[at-1])[1])
        for dtype, variant, pattern in sorted({(g['dtype'], g['variant'], g['pattern']) for g in gates}):
            own = [g for g in gates if (g['dtype'], g['variant'], g['pattern']) == (dtype, variant, pattern)]
            strict_groups.append({'dtype': dtype, 'variant': int(variant), 'pattern': pattern, 'gates': own,
                                  'pass': sum(g['pass'] == '1' for g in own), 'fail': sum(g['pass'] == '0' for g in own),
                                  'nonfinite': sum(int(g['nonfinite']) for g in own),
                                  'actual_scope': 'Dense range fallback' if all(routes[gates.index(g)] == 'dense_cublaslt' for g in own) else 'actual current-source native candidate'})
        validation['strict'] = {'path': str(path), 'exit': int(path.with_suffix('.exit').read_text()),
            'gates': len(gates), 'pass': sum(g['pass'] == '1' for g in gates), 'fail': sum(g['pass'] == '0' for g in gates),
            'nonfinite': sum(int(g['nonfinite']) for g in gates), 'native_gates': sum(route.startswith('megakernel_wgmma_psplit_rsa_bk32') for route in routes),
            'Dense_range_fallback_gates': routes.count('dense_cublaslt'), 'BF16_floor': .001, 'FP16_floor': .0001,
            'full_code_scope_count': coverage.get('strict_code_scope_not_execution', {}).get('full_code_scope_count', 1650), 'full_code_scope_executed': False,
            'scope': 'Only actual archived new candidate gates under current source. Source29 actualfull1290; source30/31/32/33 actualnew60 each; source34 actualnew36-only30, source35 actualnew38-only30 and source36 actualnew39-only30 have separate identities. Historicalfull1350/1410/1470/1530/1560/1590/1620 and currentfull1650 are unexecuted source-domain counts; source34 early1590 prediction included disabled37 and is not a historical execution.'}
    sanitizers = []
    san_record = actual.get('sanitizers')
    if san_record:
        for record in san_record['reports']:
            path = Path(record['path'])
            text = path.read_text()
            if record['tool'] == 'racecheck':
                count = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
                require(len(count) == 1, 'Actual race summary absent/ambiguous')
                hazards, errors, warnings = map(int, count[0])
            else:
                count = re.findall(r'ERROR SUMMARY: (\d+) errors', text)
                require(len(count) == 1, 'Actual sanitizer error summary absent/ambiguous')
                errors, warnings, hazards = int(count[0]), 0, 0
            leaked = None
            if record['tool'] == 'memcheck':
                count = re.findall(r'LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations', text)
                require(len(count) == 1, 'Actual memcheck leak summary absent/ambiguous')
                leaked = int(count[0][0])
            sanitizers.append({'variant': record['variant'], 'tool': record['tool'], 'path': str(path),
                               'exit': int(path.with_suffix('.exit').read_text()), 'errors': errors, 'warnings': warnings,
                               'hazards': hazards, 'leaked_bytes': leaked,
                               'test_source_sha256': plan['frozen_sources']['tests/test_correctness.cu'],
                               'test_binary_sha256': plan['frozen_binaries']['build/test_correctness'],
                               'selected_CASE': len(re.findall(r'^CASE ', text, re.M)), 'selected_PASS': len(re.findall(r'^PASS', text, re.M))})
    quick = []
    for path in sorted(Path('results').glob('small_psplit_rsa_bk32*_launch_iter37.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(rows and all((row['trials'], row['groups'], row['correct_trials'], row['quantile_unit']) == ('20', '5', '1', 'group_mean') for row in rows),
                'Quick rows use wrong declared protocol')
        quick.append({'variant': int(rows[0]['variant']), 'path': str(path), 'sha256': sha(path),
                      'exit': int(path.with_suffix('.exit').read_text()), 'rows': rows})
    compiled, compiler_audit = [], {'status': 'pending/absent actual compiled evidence input', 'gate_claimed_PASS': False}
    if compiled_path:
        record = json.loads(Path(compiled_path).read_text())
        compiled = record.get('compiled', record.get('instances', []))
        compiler_audit = record.get('compiler_audit', {'path': str(compiled_path), 'sha256': sha(compiled_path),
                                                     'status': record.get('status', 'not declared'),
                                                     'scope': 'Copied compiled review schema; status belongs to actual reviewer, no inference from register count.'})
        compiler_audit['compiled_input_path'] = str(compiled_path)
        compiler_audit['compiled_input_sha256'] = sha(compiled_path)
        compiler_audit['independent_compiled_gate'] = gate
        compiler_audit['mapped_input_scope'] = 'Mapped aliases are bound separately from the original latest compiled gate; these files need not have identical bytes.'
    environment = json.loads(Path(environment_path).read_text()) if environment_path else {
        'clocks_locked': False, 'snapshot_scope': 'See actual full CSV DEVICE and archived environment; no within-group clock evidence.'}
    omitted = {'small_psplit_rsa_bk32_profile_iter37.json', 'psplit_rsa_bk32_data_audit_iter37.json',
               'psplit_rsa_bk32_static_review_final_iter37.json', 'psplit_rsa_bk32_static_review_final_iter37.md'}
    raw = {str(path): sha(path) for path in sorted(Path('results').iterdir())
           if path.is_file() and 'iter37' in path.name and path.name not in omitted}
    if report_path and Path(report_path).exists():
        raw[str(report_path)] = sha(report_path)
    for path in Path(identity['quick_archive']).rglob('*'):
        if path.is_file():
            raw[str(path)] = sha(path)
    result = {'schema': 'megafmm.iteration37.psplit-rsa-bk32-audit.v1', 'measurement_source_commit': identity['measurement_source_commit'],
              'base_checkpoint': BASE, 'source_hashes': plan['frozen_sources'], 'binaries': plan['frozen_binaries'],
              'independent_compiled_gate': gate,
              'old_cuda_sha256': old_cuda, 'retained_old_evidence_sha256': retained,
              'retained_git_tracked_paths': sorted(retained_tracked), 'retained_onsite_only_paths': sorted(retained_onsite),
              'prior_csv_sha256': prior_csv, 'raw_sha256': raw,
              'prototype_provenance': {'scope': 'Original CPU-only RS-A prototype and models are frozen inputs, with all candidate initial quotas hypothetical until a fresh compiled source/binary gate. Final budgets/name/initial allocation come from the frozen plan and actual gate. Chronological archived build snapshots preserve their actual exit/status without retrospectively certifying pending scopes; retained unsafe37 belongs to iteration34.',
                                       'final_route_configurations': plan.get('route_configurations', {}),
                                       'CPU_models': [validation['CPU_state_model']] if validation['CPU_state_model'] else []},
              'environment': environment, 'interface': {'new_variants': [40], 'raw_configuration_mapping': {'40': 0},
                   'new_variants_scope': 'Declared source IDs; actual allowed native execution follows final compiled safety and public API support. Missing IDs are not native-certified.',
                   'new_variants_actually_measured': sorted({int(row['csv_row']['variant']) for row in metrics['rows']
                                                           if row['csv_row']['requested'] == 'mega' and row['csv_row']['variant'] == '40'}),
                   'explicit_unmeasured_keys': metrics['measurement_scope']['missing_rows'],
                   'final_route_configurations': plan.get('route_configurations', {}), 'maximum_valid_megakernel_variant': 40,
                   'maximum_registered_variant': 40,
                   'maximum_enabled_native_variant': max([39] + [int(key) for key, value in plan.get('route_configurations', {}).items() if value.get('enabled') is True]),
                   'disabled_native_variants': sorted({37} | {int(key) for key, value in plan.get('route_configurations', {}).items() if value.get('enabled') is False}),
                   'Auto': 'Dense', 'native_workspace_bytes': 0, 'ordinary_native_complete_kernel_count': 1,
                   'precision': 'Original FP32 RN S/T DAG, final onecast, complete wholeK seven FP32P, fixedU, explicit alphaRN and activebetaFMA; originalroundedAB FP64gates unchanged.',
                   'consumer_schedule': 'New40 BK32 RS-A/raw A4 shared and final B7 shared in two banks: consumers form distinct whole-stage register A fragments and perform ascending wholeK P3+4, eachwait0 before generic reuse/release. Final source drain/barrier/role allocations and allsevenP fixedU/onecast epilogue follow the frozen source and independent fresh compiled gate. Current native K guard and range fallback require actually archived current-source checks.', 'geometry_scope': 'BM64 BN32 BK32 Stages2 Threads384 RS-A new configuration; final quotas/name/minCTAs/actualshared and initialallocation are read from the frozen plan and fresh source/object/bench/test proof. CPU shared/register hypotheses are not actual compiled fit. No inherited36 launch gate or dynamicoccupancy/liveness certification.',
                   'range_fallback_scope': 'Dense fallback PASS is not native numerical certification.'},
              'validation': validation, 'strict_groups': strict_groups, 'sanitizers': sanitizers,
              'quick_protocol': {'warmup': 20, 'calls_per_group': 20, 'groups': 5, 'fresh_checks_per_row': 1, 'rotate_inputs': 1,
                                 'workspace_limit': 8589934592, 'heuristic_candidates': 32, 'quantile_unit': 'group_mean', 'screen_only': True},
              'quick_measurements': quick, 'full1024': {'plan_path': str(plan_path), 'measurements_path': str(measurements_path),
                   'warmup': 20, 'calls_per_group': 100, 'groups': 5, 'fresh_checks_per_row': 3,
                   **{field: metrics[field] for field in ('row_count', 'fresh_gate_count', 'fresh_gate_pass', 'fresh_gate_fail', 'nonfinite', 'measurement_scope',
                     'same_binary_megakernel_comparison', 'new_vs_ss_bk32_comparison', 'each_new_vs_retained_controls', 'RS_A_vs_SS_BK32_axis_comparison', 'RS_A_BK_axis_comparison', 'observed_all_sw_route_winners')},
                        'batchwall_protocol': metrics['batchwall_protocol'], 'same_row_dense_over_sw': 'Independently tuned same-row cuBLASLt eventmedian / completeSW eventmedian; Dense and SW timed in the same launch/Graph mode, with tuning before capture and no Graph-specific tuning mode.'},
              'compiled': compiled, 'compiler_audit': compiler_audit,
              'independent_data_audit': {'agent': 'pipeline_data30', 'raw_status': 'pending independent audit', 'final_profile_status': 'pending readonly audit'},
              'independent_static_final_audit': {'agent': 'pipeline_review30', 'status': 'pending final review'},
              'interpretation': ['Rank new40 own SW latency vs samebinary39(previous BK64 RS-A best) and36(matched BK32 SS control) only when actually measured. BK-axis and RS-A/SS observations are not isolated hardware-cause claims.',
                                 'Partial compiled-blocked rows and pending validation are explicit; expected16 is a measurement target and full1650 is unexecuted current code scope, not actual execution.',
                                 'Original numerical gate/floor/AutoDense remain unchanged; no numerical certification from structural PASS.',
                                 'Lt1 is complete multi-launch control, not singlelaunch goal completion.',
                                 'No measured occupancy, physicalfloor or isolatedstall/traffic causal claim.']}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='results/psplit_rsa_bk321024_plan_iter37.json')
    parser.add_argument('--identity', default='results/psplit_rsa_bk32_measured_identity_iter37.json')
    parser.add_argument('--measurements', default='results/small_psplit_rsa_bk32_measurements_iter37.json')
    parser.add_argument('--coverage', default='results/psplit_rsa_bk32_test_coverage_iter37.json')
    parser.add_argument('--compiled-json', help='Actual mapped compiled records: {compiled:[...],compiler_audit:{...}} or actual review instances')
    parser.add_argument('--environment-json', help='Actual current environment metadata, never inherited as current evidence')
    parser.add_argument('--report', default='results/1024_MEGAKERNEL_PSPLIT_RSA_BK32_REPORT.md')
    parser.add_argument('--output', help='Owner-selected formal output; default stdout')
    args = parser.parse_args()
    value = generate(args.plan, args.identity, args.measurements, args.coverage, args.compiled_json, args.environment_json, args.report)
    text = json.dumps(value, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')
