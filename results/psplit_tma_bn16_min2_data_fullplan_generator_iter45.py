#!/usr/bin/env python3
"""Owner-run one-time iteration45 full-plan freeze; default stdout only.

The archived iteration44 plan supplies the timing protocol only. Current names,
quotas, resources and enabled scope must be passed explicitly and bound by the
fresh iteration45 compiled certificate. Initial pending history is preserved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

TEMPLATE = 'results/psplit_tma_bn161024_plan_iter44.json'
SOURCES = {'CMakeLists.txt', 'bench/benchmark.cu', 'include/megafmm/gemm.h',
           'include/megafmm/kernels.h', 'solution/plan.cpp', 'tests/test_correctness.cu',
           'solution/megakernel_wgmma_psplit_tma_bn16_min2.cu'}
BINS = {'build/megafmm_bench', 'build/test_correctness'}
ROUTES = [('psplit_tma_bn16_min2_49','mega',49),('psplit_tma_bn16_48','mega',48),('psplit_selective_tma_a45','mega',45),('psplit_rsa39','mega',39),('packed_lt1','lt',1)]
PROTOCOL = ('shapes', 'dtypes', 'modes', 'warmup', 'trials', 'groups',
            'correct_trials', 'rotate_inputs', 'heuristic_candidates', 'workspace_limit', 'reference')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(path), 'sha256': sha(path)}


def protocol(plan):
    require(plan['shapes'] == [[1024, 1024, 1024]] and plan['dtypes'] == ['bf16', 'fp16'] and
            plan['modes'] == ['launch', 'graph'], 'Require the archived mandatory1024/two-dtype/two-mode domain')
    require(tuple(plan[k] for k in PROTOCOL[3:-1]) == (20, 100, 5, 3, 1, 32, 8589934592),
            'Require unchanged full20w/100calls/5groups/3fresh/rotate1/32Lt/8GiB protocol')


def verify_inputs(gate_path, initial_path, configurations):
    initial, gate = read(initial_path), read(gate_path)
    require(set(initial['frozen_sources']) == SOURCES and set(initial['frozen_binaries']) == BINS,
            'Require exact current seven-source/two-binary identity')
    freeze = {**initial['frozen_sources'], **initial['frozen_binaries']}
    for path, wanted in freeze.items():
        require(sha(path) == wanted, 'Current frozen bytes differ from initial identity: ' + path)
    require('iter45' in gate.get('schema', '') and 'compiled' in gate.get('schema', ''),
            'Require an original iteration45 compiled gate')
    fresh_flags={k:v for k,v in gate.items() if k.startswith('compiled_gate_inherited_from_')}
    require(fresh_flags and all(v is False for v in fresh_flags.values()) and
            (fresh_flags.get('compiled_gate_inherited_from_iteration44') is False or
             any(('any_prior' in k or 'older_iterations' in k or 'prior_iterations' in k) for k in fresh_flags)),
            'Require literal fresh current45 gate flags; no inherited parent or older proof')
    status = gate.get('status')
    require(isinstance(status, str) and status and not any(x in status.upper() for x in ('PENDING', 'UNKNOWN')),
            'Fresh gate must be terminal; initial pending history cannot certify current scope')
    for path, wanted in freeze.items():
        require(gate.get('frozen_bindings', {}).get(path) == wanted, 'Fresh gate freeze mismatch: ' + path)
    link_path, link_sha = initial['linked_identity_path'], initial['linked_identity_sha256']
    require(sha(link_path) == link_sha and gate['frozen_bindings'].get(link_path) == link_sha,
            'Current linked identity must match initial freeze and fresh gate')
    linked = read(link_path)
    require(all(linked.get(path) == wanted for path, wanted in initial['frozen_binaries'].items()),
            'Linked identity must bind both current binaries')
    require(set(configurations) == {'49','45','48','39'},
            'Explicit configurations for exactly49/45/48/39 are required')
    for key, config in configurations.items():
        require(isinstance(config.get('enabled'), bool), 'Final enabled must be explicit bool: ' + key)
        require(config.get('actual_name') and all(config.get(k) is not None for k in
                ('producer_quota', 'launch_bounds_min_ctas', 'BM', 'BN', 'BK', 'threads')) and
                len(config.get('consumer_quotas', [])) == 2 and all(q is not None for q in config['consumer_quotas']),
                'Require explicit actual name, geometry, role quotas and launch bounds: ' + key)
        require(all(k in config for k in ('dynamic_shared_bytes', 'static_shared_bytes')),
                'Shared fields must explicitly represent actual allocation or blocked unknown null: ' + key)
        require(not config['enabled'] or all(config[k] is not None for k in ('dynamic_shared_bytes', 'static_shared_bytes')),
                'Enabled route requires actual shared allocation: ' + key)
        variant = int(key)
        if variant != 49:
            require(config['enabled'] is True, 'All retained controls remain measured')
            continue
        safe = [x for x in gate.get('instances', []) if x.get('variant') == variant]
        blocked = [x for x in gate.get('disabled_instances', []) if x.get('variant') == variant]
        entries = safe if config['enabled'] else blocked
        require(len(entries) == 2 and {str(x.get('dtype', '')).lower() for x in entries} == {'bf16', 'fp16'},
                'Fresh gate must contain both actual dtype instances for each enabled/blocked route: ' + key)
        require(not (blocked if config['enabled'] else safe), 'Fresh gate scope conflicts with route enabled: ' + key)
        for instance in entries:
            require(str(instance.get('status', '')).startswith('PASS' if config['enabled'] else 'FAIL'),
                    'Fresh instance status must match enabled scope: ' + key)
            if config['enabled']:
                require(instance.get('producer_quota') == config['producer_quota'] and
                        instance.get('consumer_quotas') == config['consumer_quotas'] and
                        instance.get('dynamic_shared_bytes') == config['dynamic_shared_bytes'] and
                        instance.get('actual_driver_static_shared_bytes',instance.get('static_shared_bytes')) == config['static_shared_bytes'] and
                        instance.get('bm') == config['BM'] and instance.get('bn') == config['BN'] and
                        instance.get('bk') == config['BK'] and instance.get('threads') == config['threads'],
                        'Actual gate quotas/shared/geometry must agree with passed route configurations: ' + key)
    return initial, {**binding(gate_path), 'status': status}


def ensure_before_timing(plan):
    for path in plan['csv_paths']:
        for suffix in ('.csv', '.log', '.exit'):
            require(not Path(path).with_suffix(suffix).exists(), 'Full timing artifact already exists: ' + path)


def generate(gate_path, initial_path, configuration_path, template_path=TEMPLATE):
    configurations = read(configuration_path)
    initial, gate = verify_inputs(gate_path, initial_path, configurations)
    template = read(template_path); protocol(template)
    plan = {key: template[key] for key in PROTOCOL}
    plan.update(schema='megafmm.iter45.full1024-plan.v1', declared_before_measurement=True,
        reason='Mandatory new BN16 min2 TMA-A49 versus retained BN16 TMA-A48, BN32 TMA-A45, RS-A39 and Lt1 full20. Current-A encode consumed by TMA inside enabled ordinary49/48/45 complete calls; compiled blocks retain exact missing keys irrespective quick outcome.',
        routes=[{'label': label, 'algorithm': alg, 'variant': variant} for label, alg, variant in ROUTES],
        frozen_sources=initial['frozen_sources'], frozen_binaries=initial['frozen_binaries'],
        route_configurations=configurations, independent_compiled_gate=gate,
        initial_identity=binding(initial_path), route_configuration_input=binding(configuration_path),
        csv_paths=[f'results/small_{label}_{mode}_iter45_full1024.csv' for label, _, _ in ROUTES for mode in plan['modes']],
        timing_contract='Complete invocation including all input transforms/products/output epilogue. Independent same-row cuBLASLt tuning before capture, no Graph-specific tuning. Ordinary49,48 and45 each include one current-A tensor-map encode plus the complete kernel;49,48 and45 consume their current A descriptor for TMA rawA. Control39 encodes no descriptor. Graph encodes the A descriptor during capture for fixed pointers and replays fresh contents without re-encoding. Ordinary/Graph each20warmup/100calls/5groups/3fresh; full GPU event and synchronized100-call batch wall/100, tune/setup/capture excluded. Original FP64 gates/floors unchanged.',
        freeze_status='Fresh terminal iteration45 gate bound before mandatory timing; actual enabled scope comes from current route configurations. Initial pending snapshot remains historical; compiled safety is separate from numerical/performance certification.',
        expected_row_count=len(ROUTES) * len(plan['dtypes']) * len(plan['modes']), enabled_row_count=4 * (4 + int(configurations['49']['enabled'])))
    plan['new_variants'] = [49]
    plan['new_source_paths'] = ['solution/megakernel_wgmma_psplit_tma_bn16_min2.cu']
    plan['comparison_pairs'] = [{'candidate_algorithm':'mega', 'candidate_variant':candidate,
        'baseline_algorithm':'mega', 'baseline_variant':baseline, 'axis':f'BN16_MIN2_CONTROL{candidate}_vs_{baseline}',
        'controlled_axis_claimed':False} for candidate, baseline in ((49,48),(49,45),(45,39),(49,39))]
    plan['retained_control_resource_scope'] = 'Legacy controls retain previously frozen attributes; these are not new iteration45 function resource or residency queries.'
    for route in plan['routes']:
        route['actual_name'] = configurations[str(route['variant'])]['actual_name'] if route['algorithm']=='mega' else 'winograd_lt_batched'
        route['enabled'] = configurations[str(route['variant'])]['enabled'] if route['algorithm']=='mega' else True
        route['native'] = route['algorithm']=='mega'
    plan['compiled_blocked_rows'] = [{'algorithm': 'mega', 'variant': str(v), 'dtype': dtype, 'mode': mode}
        for v in (49,) if not configurations[str(v)]['enabled'] for dtype in plan['dtypes'] for mode in plan['modes']]
    ensure_before_timing(plan)
    return plan


def emit(value, output):
    content = json.dumps(value, indent=2) + '\n'
    if output:
        with Path(output).open('x') as stream:
            stream.write(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('gate', 'initial-identity', 'route-configurations'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--template', default=TEMPLATE)
    parser.add_argument('--output', help='Owner-selected one-time formal path; default stdout')
    args = parser.parse_args()
    emit(generate(args.gate, args.initial_identity, args.route_configurations, args.template), args.output)
