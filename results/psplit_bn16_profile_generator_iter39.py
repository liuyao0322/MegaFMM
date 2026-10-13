#!/usr/bin/env python3
"""Owner-run dynamic iteration39 evidence profile generator.

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

BASE = 'd0ea861605612d0e4d95e2e6596ff352a2d264dd'
PRIOR = 'results/small_psplit_hybrid_s_profile_iter38.json'
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
            'Require the actual current iteration39 independent compiled gate')
    require(metrics.get('independent_compiled_gate') == gate,
            'Metrics and identity must bind the same latest original compiled gate')
    require(sha(gate['path']) == gate['sha256'], 'Original compiled gate onsite SHA mismatch')
    require(not any(word in str(gate['status']).upper() for word in ('PENDING', 'UNKNOWN')),
            'The original compiled gate status must not be pending')
    original = json.loads(Path(gate['path']).read_text())
    require('iter39' in original.get('schema', '') and 'compiled' in original.get('schema', ''),
            'Require an iteration39 original compiled review schema')
    require(original.get('status') == gate['status'], 'Original gate and identity status mismatch')
    require(original.get('compiled_gate_inherited_from_iteration38') is False,
            'Original gate must explicitly declare no inherited iteration38 certification')
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
    require(len(plan['frozen_sources']) == 8 and len(plan['frozen_binaries']) == 2, 'Require actual eight-source/two-binary freeze')
    expected_sources = {'solution/megakernel_wgmma_psplit_rsa_bn16.cu', 'solution/megakernel_wgmma_psplit_ss_bn16.cu', 'solution/plan.cpp',
                        'include/megafmm/kernels.h', 'include/megafmm/gemm.h', 'CMakeLists.txt',
                        'bench/benchmark.cu', 'tests/test_correctness.cu'}
    require(set(plan['frozen_sources']) == expected_sources, 'Require all six common files and the two distinct BN16 TUs')
    require(tuple(plan[key] for key in ('warmup', 'trials', 'groups', 'correct_trials', 'rotate_inputs',
                                       'heuristic_candidates', 'workspace_limit')) == (20, 100, 5, 3, 1, 32, 8589934592),
            'Mandatory full measurement protocol changed')
    require(metrics['measurement_scope']['expected_row_count'] == 24, 'Six routes/two dtypes/two timing modes target24')
    require(identity['measurement_source_commit'] == metrics['measurement_source_commit'], 'Metrics measured-source mismatch')
    require(metrics['frozen_binaries'] == plan['frozen_binaries'], 'Metrics frozen-binary mismatch')
    require(metrics.get('frozen_sources') == plan['frozen_sources'], 'Metrics frozen-source mismatch')
    require(metrics['predeclared_plan'] == {'path': str(plan_path), 'sha256': sha(plan_path)}, 'Metrics plan identity mismatch')
    require(metrics.get('route_configurations', {}) == plan.get('route_configurations', {}), 'Metrics final route configurations mismatch')
    require(all(isinstance(plan.get('route_configurations', {}).get(str(v), {}).get('enabled'), bool) for v in (42, 43)),
            'Final42/43 enabled must be explicitly frozen from fresh compiled gate and public API scope')
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
    prior_metrics = base_json('results/small_psplit_hybrid_s_measurements_iter38.json')
    old_cuda = {}
    for path in sorted(Path('solution').glob('*.cu')):
        if str(path) in plan['frozen_sources']:
            continue
        data = subprocess.check_output(['git', 'show', f'{BASE}:{path}'])
        require(path.read_bytes() == data, 'Old CUDA source drift: ' + str(path))
        old_cuda[str(path)] = sha(path)
    require(len(old_cuda) == 35, 'Require all35 old CUDA sources')
    retained = {**prior['retained_old_evidence_sha256'], **prior['raw_sha256']}
    extra = [PRIOR, 'results/small_psplit_hybrid_s_measurements_iter38.json', 'results/1024_MEGAKERNEL_PSPLIT_HYBRID_S_REPORT.md',
             'results/strict_psplit_hybrid_s_iter38.txt', 'results/strict_psplit_hybrid_s_iter38.exit',
             'results/psplit_hybrid_s_test_coverage_iter38.json', 'results/psplit_hybrid_s_static_review_final_iter38.json',
             'results/psplit_hybrid_s_static_review_final_iter38.md', 'results/psplit_hybrid_s_data_audit_iter38.json']
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
                  'disabled37_cli': actual_log(actual.get('disabled37_cli')),
                  'disabled_policy_coverage_path': str(coverage_path),
                  'test_coverage_audit_path': str(coverage_path), 'strict': None, 'strict_by_variant': {},
                  'execution_scope': 'Only archived current-source runs are actual; blocked or absent scopes stay null. Current full code domain is 1680 plus30 for each enabled new variant (both1740/one1710/none1680); it is not an executed full rerun. Historical scopes remain separately bound.'}
    for variant in (42, 43):
        validation['selected'+str(variant)] = actual_log(actual.get('selected'+str(variant)))
        validation['disabled'+str(variant)+'_cli'] = actual_log(actual.get('disabled'+str(variant)+'_cli'))
    validation['CPU_checks'] = {}
    for label in ('DAG', 'stage'):
        path = Path(f'results/psplit_bn16_cpu_{label.lower()}_iter39.txt')
        if not path.exists():
            validation['CPU_checks'][label+'_scope'] = 'Actual current archived run absent; no PASS or exit inferred.'
            continue
        counts = re.findall(r'Ran (\d+) tests', path.read_text())
        require(len(counts) == 1, 'Actual CPU unittest count absent/ambiguous: '+str(path))
        validation['CPU_checks'].update({label+'_path': str(path), label+'_tests': int(counts[0]),
                                         label+'_exit': int(path.with_suffix('.exit').read_text())})
    for stem in ('raw_model', 'ss_model', 'tail_model'):
        path = Path(f'results/psplit_bn16_cpu_{stem}_iter39.txt')
        validation['CPU_'+stem] = None
        if path.exists():
            value = stdout_object(path)
            validation['CPU_'+stem] = {'path': str(path), 'exit': int(path.with_suffix('.exit').read_text()),
                'exit_recorded': True, 'exit_path': str(path.with_suffix('.exit')), 'status': value.get('status'),
                'stdout_summary': value,
                'scope': 'Actual protected frozen BN16 source/abstract/literal-coordinate model stdout and archived process exit. CPU scalar bank sets do not prove actual transaction grouping, stall rate, register fit, two-CTA residency, numerical GEMM certification or GPU performance. Frozen input artifacts and historical output writers remain protected.',
                'not_hardware_liveness_certificate': True}
    validation['new_by_variant'] = {str(v): None for v in (42, 43)}
    default = validation['default']
    if default:
        lines = Path(default['path']).read_text().splitlines()
        all_graphs = [fields(line) for line in lines if line.startswith('PASS') and 'GRAPH' in line.split()[0] and 'replays=' in line]
        for variant in (42, 43):
            v = str(variant)
            graphs = [entry for entry in all_graphs if entry.get('variant') == v]
            bindings = [fields(line) for line in lines if line.startswith('PASS') and 'INPUT_REBIND' in line.split()[0] and fields(line).get('variant') == v]
            bits = [fields(line) for line in lines if line.startswith('PASS') and ('NATIVE_BITWISE' in line.split()[0] or 'CROSSATOM_BITWISE' in line.split()[0]) and fields(line).get('variant') == v]
            same_bits = [fields(line) for line in lines if line.startswith('PASS') and 'NATIVE_BITWISE' in line.split()[0] and fields(line).get('variant') == v]
            cross_bits = [fields(line) for line in lines if line.startswith('PASS') and 'CROSSATOM_BITWISE' in line.split()[0] and fields(line).get('variant') == v]
            validation['new_by_variant'][v] = {'new_graph_configurations': len(graphs),
                'new_graph_fresh_replays': sum(int(entry['replays']) for entry in graphs),
                'sameatom_rawU_host_epilogue_groups': len(same_bits),
                'crossatom_rawU_host_epilogue_groups': len(cross_bits),
                'bitwise_rawU_host_epilogue_groups': len(bits),
                'current_pointer_binding_checks': sum(int(entry['bindings']) for entry in bindings),
                'enabled': plan['route_configurations'][v]['enabled'],
                'reference_scope': '42 uses43 N16 sameatom reference only when43 is enabled, otherwise42 uses36 as crossatom N16-vs-N32;43 always uses36 as explicit crossatom N16-vs-N32 check. Reference identity, actual fixture multiplicity and any FAIL come from frozen test source/coverage; no N32 oracle relabeled N16.'}
        validation.update({'new_graph_configurations': sum(x['new_graph_configurations'] for x in validation['new_by_variant'].values()),
            'new_graph_fresh_replays': sum(x['new_graph_fresh_replays'] for x in validation['new_by_variant'].values()),
            'total_graph_configurations': len(all_graphs), 'total_graph_fresh_replays': sum(int(entry['replays']) for entry in all_graphs),
            'sameatom_rawU_host_epilogue_groups': sum(x['sameatom_rawU_host_epilogue_groups'] for x in validation['new_by_variant'].values()),
            'crossatom_rawU_host_epilogue_groups': sum(x['crossatom_rawU_host_epilogue_groups'] for x in validation['new_by_variant'].values()),
            'bitwise_rawU_host_epilogue_groups': sum(x['bitwise_rawU_host_epilogue_groups'] for x in validation['new_by_variant'].values()),
            'current_pointer_binding_checks': sum(x['current_pointer_binding_checks'] for x in validation['new_by_variant'].values()),
            'sameatom_scope': 'Actual newvariant fixture loops retain declared reference identities:42-to43 N16 only when enabled, otherwise42-to36 crossatom;43-to36 is always crossatom N16-vs-N32. Each uses independent host RNalpha/betaFMA. Disabled or unexecuted routes are never native certified.'})
    else:
        validation.update({field: None for field in ('new_graph_configurations', 'new_graph_fresh_replays', 'total_graph_configurations',
                          'total_graph_fresh_replays', 'sameatom_rawU_host_epilogue_groups', 'crossatom_rawU_host_epilogue_groups', 'bitwise_rawU_host_epilogue_groups', 'current_pointer_binding_checks')})
    strict_groups, all_strict_gates, all_strict_routes, strict_paths = [], [], [], []
    # Actual coverage adapters may use one aggregate strict record or separate variant records.
    strict_records = [(label, record) for label, record in actual.items() if label.startswith('strict_') and isinstance(record, dict) and record.get('path')]
    for label, record in strict_records:
        path = Path(record['path']); lines = path.read_text().splitlines(); gates, routes = [], []
        for at, line in enumerate(lines):
            if line.startswith('NUMERICAL_GATE '):
                gates.append(fields(line)); routes.append(re.search(r'actual=(\S+)', lines[at-1])[1])
        variants = {int(g['variant']) for g in gates}
        require(variants <= {42,43}, 'Current strict archive must concern only new42/new43')
        own_stats = {'path': str(path), 'exit': int(path.with_suffix('.exit').read_text()),
            'gates': len(gates), 'pass': sum(g['pass']=='1' for g in gates), 'fail': sum(g['pass']=='0' for g in gates),
            'nonfinite': sum(int(g['nonfinite']) for g in gates),
            'native_gates': sum(route in {plan['route_configurations'][str(v)]['actual_name'] for v in variants} for route in routes),
            'Dense_range_fallback_gates': routes.count('dense_cublaslt'), 'variants': sorted(variants),
            'BF16_floor': .001, 'FP16_floor': .0001}
        for variant in variants:
            own = [g for g in gates if int(g['variant'])==variant]
            own_routes = [routes[i] for i,g in enumerate(gates) if int(g['variant'])==variant]
            validation['strict_by_variant'][str(variant)] = {'record_label': label, **own_stats,
                'gates': len(own), 'pass': sum(g['pass']=='1' for g in own), 'fail': sum(g['pass']=='0' for g in own),
                'nonfinite': sum(int(g['nonfinite']) for g in own), 'variants':[variant],
                'native_gates': sum(route==plan['route_configurations'][str(variant)]['actual_name'] for route in own_routes),
                'Dense_range_fallback_gates': own_routes.count('dense_cublaslt')}
        for dtype, variant, pattern in sorted({(g['dtype'],g['variant'],g['pattern']) for g in gates}):
            indexes = [i for i,g in enumerate(gates) if (g['dtype'],g['variant'],g['pattern'])==(dtype,variant,pattern)]
            own = [gates[i] for i in indexes]
            strict_groups.append({'dtype':dtype,'variant':int(variant),'pattern':pattern,'gates':own,
                'pass':sum(g['pass']=='1' for g in own),'fail':sum(g['pass']=='0' for g in own),'nonfinite':sum(int(g['nonfinite']) for g in own),
                'actual_scope':'Dense range fallback' if all(routes[i]=='dense_cublaslt' for i in indexes) else 'actual current-source native candidate'})
        all_strict_gates.extend(gates); all_strict_routes.extend(routes); strict_paths.append(own_stats)
    if strict_paths:
        validation['strict'] = {'paths': strict_paths, 'gates': len(all_strict_gates),
            'pass': sum(g['pass']=='1' for g in all_strict_gates), 'fail': sum(g['pass']=='0' for g in all_strict_gates),
            'nonfinite': sum(int(g['nonfinite']) for g in all_strict_gates),
            'native_gates': sum(route in {plan['route_configurations'][str(v)]['actual_name'] for v in (42,43)} for route in all_strict_routes),
            'Dense_range_fallback_gates': all_strict_routes.count('dense_cublaslt'), 'BF16_floor':.001,'FP16_floor':.0001,
            'full_code_scope_count': coverage.get('strict_code_scope_not_execution', {}).get('full_code_scope_count',1680+30*sum(plan['route_configurations'][str(v)]['enabled'] for v in (42,43))),
            'full_code_scope_executed': False,
            'scope': 'Only actual new42/new43 archived strict runs. Historical actualfull1290 and separately sourced new60/new30 runs remain history. Current full1680/1710/1740 is enabled-variant code domain, not executed full rerun; Dense range fallback PASS does not certify native.'}
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
    for path in sorted(Path('results').glob('small_psplit_bn16*_launch_iter39.csv')):
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
    omitted = {'small_psplit_bn16_profile_iter39.json', 'psplit_bn16_data_audit_iter39.json',
               'psplit_bn16_static_review_final_iter39.json', 'psplit_bn16_static_review_final_iter39.md'}
    raw = {str(path): sha(path) for path in sorted(Path('results').iterdir())
           if path.is_file() and 'iter39' in path.name and path.name not in omitted}
    if report_path and Path(report_path).exists():
        raw[str(report_path)] = sha(report_path)
    for path in Path(identity['quick_archive']).rglob('*'):
        if path.is_file():
            raw[str(path)] = sha(path)
    result = {'schema': 'megafmm.iteration39.psplit-bn16-audit.v1', 'measurement_source_commit': identity['measurement_source_commit'],
              'base_checkpoint': BASE, 'source_hashes': plan['frozen_sources'], 'binaries': plan['frozen_binaries'],
              'independent_compiled_gate': gate,
              'old_cuda_sha256': old_cuda, 'retained_old_evidence_sha256': retained,
              'retained_git_tracked_paths': sorted(retained_tracked), 'retained_onsite_only_paths': sorted(retained_onsite),
              'prior_csv_sha256': prior_csv, 'raw_sha256': raw,
              'prototype_provenance': {'scope': 'Frozen BN16 raw/SS candidates, tail-layout derivatives and three manifests remain CPU-only inputs. Each newroute requires its own fresh source/binary compiled safety proof; final budgets/name/shared/initial ownership come from the frozen plan and actual gate. Chronological build attempts preserve actual exits; no pending scope is retrospectively certified and retained unsafe37 belongs to iteration34.',
                                       'final_route_configurations': plan.get('route_configurations', {}),
                                       'CPU_models': [validation['CPU_'+key] for key in ('raw_model','ss_model','tail_model') if validation['CPU_'+key]]},
              'environment': environment, 'interface': {'new_variants': [42,43], 'raw_configuration_mapping': {'42':0,'43':0},
                   'new_variants_scope': 'Declared source IDs; actual allowed native execution follows final compiled safety and public API support. Missing IDs are not native-certified.',
                   'new_variants_actually_measured': sorted({int(row['csv_row']['variant']) for row in metrics['rows']
                                                           if row['csv_row']['requested'] == 'mega' and row['csv_row']['variant'] in ('42','43')}),
                   'explicit_unmeasured_keys': metrics['measurement_scope']['missing_rows'],
                   'final_route_configurations': plan.get('route_configurations', {}), 'maximum_valid_megakernel_variant':43,
                   'maximum_registered_variant':43,
                   'maximum_enabled_native_variant': max([41] + [int(key) for key, value in plan.get('route_configurations', {}).items() if value.get('enabled') is True]),
                   'disabled_native_variants': sorted({37} | {int(key) for key, value in plan.get('route_configurations', {}).items() if value.get('enabled') is False}),
                   'Auto': 'Dense', 'native_workspace_bytes': 0, 'ordinary_native_complete_kernel_count': 1,
                   'precision': 'Original FP32 RN S/T DAG, final onecast, complete wholeK seven FP32P, fixedU, explicit alphaRN and activebetaFMA; originalroundedAB FP64gates unchanged.',
                   'consumer_schedule': 'New42 raw RS-A BN16 and new43 shared-A SS BN16 retain the original FP32 RN S/T DAG and final onecast, wholeK P3+4 ownership, seven FP32 products, fixedU and RNalpha/activebetaFMA epilogue. Final source and fresh compiled gate define role quotas, stage lifetime, drain/barrier, typed ABI and tail mapping. The BN16 scalar CPU bank model is not actual instruction grouping or traffic.',
                   'geometry_scope': 'Final BM64/BN16/BK32/Stages2/Threads384 new geometries and minCTAs/quotas/name/actualshared are read from frozen plan and perroute fresh object/bench/test proof. BN, quota, launch bounds and tail mapping differ together from controls; static minCTA annotation is not measured two-CTA residency. No inherited39/40/41 gate or dynamicoccupancy certification.',
                   'range_fallback_scope': 'Dense fallback PASS is not native numerical certification.'},
              'validation': validation, 'strict_groups': strict_groups, 'sanitizers': sanitizers,
              'quick_protocol': {'warmup': 20, 'calls_per_group': 20, 'groups': 5, 'fresh_checks_per_row': 1, 'rotate_inputs': 1,
                                 'workspace_limit': 8589934592, 'heuristic_candidates': 32, 'quantile_unit': 'group_mean', 'screen_only': True},
              'quick_measurements': quick, 'full1024': {'plan_path': str(plan_path), 'measurements_path': str(measurements_path),
                   'warmup': 20, 'calls_per_group': 100, 'groups': 5, 'fresh_checks_per_row': 3,
                   **{field: metrics[field] for field in ('row_count', 'fresh_gate_count', 'fresh_gate_pass', 'fresh_gate_fail', 'nonfinite', 'measurement_scope',
                     'same_binary_megakernel_comparison','new_raw_bn16_vs_raw_bn32_comparison','new_ss_bn16_vs_ss_bn32_comparison',
                     'new_raw_bn16_vs_ss_bn16_comparison','each_new_vs_retained_controls','same_binary_own_latency_comparison',
                     'own_latency_pair_contract','axis_comparisons','observed_all_sw_route_winners','observed_native_sw_route_winners')},
                        'batchwall_protocol': metrics['batchwall_protocol'], 'same_row_dense_over_sw': 'Independently tuned same-row cuBLASLt eventmedian / completeSW eventmedian; Dense and SW timed in the same launch/Graph mode, with tuning before capture and no Graph-specific tuning mode.'},
              'compiled': compiled, 'compiler_audit': compiler_audit,
              'independent_data_audit': {'agent': 'pipeline_data30', 'raw_status': 'pending independent audit', 'final_profile_status': 'pending readonly audit'},
              'independent_static_final_audit': {'agent': 'pipeline_review30', 'status': 'pending final review'},
              'interpretation': ['Rank actual new42/new43 own whole-call SW latency in the same fresh binary. The unique five pairs observe42vs39/40/43 and43vs39/36 separately; allcontrolled_axis_claimed=false. Dense denominator fluctuations do not prove code benefits.',
                                 'Expected24 is mandatory target; compiled-blocked rows are exact explicit missing keys, not measurements. Current strictfull1680/1710/1740 is code domain only unless separately executed; no historical measurements substitute for this binary.',
                                 'Original numerical gate/floor/AutoDense remain unchanged; structural PASS does not certify precision or complete the goal.',
                                 'Lt1 is complete multi-launch control, not singlelaunch goal completion.',
                                 'CPU scalar bank sets and launch_bounds(min2) do not establish actual transactions, bankstalls or measured two-CTA residency. No occupancy, physicalfloor or isolated causal claim.']}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='results/psplit_bn161024_plan_iter39.json')
    parser.add_argument('--identity', default='results/psplit_bn16_measured_identity_iter39.json')
    parser.add_argument('--measurements', default='results/small_psplit_bn16_measurements_iter39.json')
    parser.add_argument('--coverage', default='results/psplit_bn16_test_coverage_iter39.json')
    parser.add_argument('--compiled-json', help='Actual mapped compiled records: {compiled:[...],compiler_audit:{...}} or actual review instances')
    parser.add_argument('--environment-json', help='Actual current environment metadata, never inherited as current evidence')
    parser.add_argument('--report', default='results/1024_MEGAKERNEL_PSPLIT_BN16_REPORT.md')
    parser.add_argument('--output', help='Owner-selected formal output; default stdout')
    args = parser.parse_args()
    value = generate(args.plan, args.identity, args.measurements, args.coverage, args.compiled_json, args.environment_json, args.report)
    text = json.dumps(value, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')
