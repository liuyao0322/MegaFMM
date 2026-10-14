#!/usr/bin/env python3
"""Owner-run complete measured identity after quicklogger/Git, before full timing."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True


def helpers():
    path = Path(__file__).with_name('psplit_tma_bn64_data_fullplan_generator_iter43.py')
    spec = importlib.util.spec_from_file_location('sw128_simt_control_fullplan42', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def generate(plan_path, gate_path, initial_path, configuration_path, quick_archive):
    h = helpers()
    initial, gate = h.verify_inputs(gate_path, initial_path, h.read(configuration_path))
    plan = h.read(plan_path); h.protocol(plan); h.ensure_before_timing(plan)
    h.require(plan['schema'] == 'megafmm.iter43.full1024-plan.v1' and plan['expected_row_count'] == 20,
              'Require the already frozen mandatory20 plan')
    h.require(plan['frozen_sources'] == initial['frozen_sources'] and plan['frozen_binaries'] == initial['frozen_binaries'],
              'Plan and current initial freeze must match')
    h.require(plan['independent_compiled_gate'] == gate and plan['route_configurations'] == h.read(configuration_path),
              'Plan must bind latest gate and exact passed configurations')
    h.require(plan['initial_identity'] == h.binding(initial_path) and plan['route_configuration_input'] == h.binding(configuration_path),
              'Plan input bytes must remain frozen')
    archive = Path(quick_archive)
    h.require((archive / 'results.csv').is_file(), 'Require the owner-closed AKO quicklogger archive')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for path, wanted in initial['frozen_sources'].items():
        h.require(hashlib.sha256(subprocess.check_output(['git', 'show', head + ':' + path])).hexdigest() == wanted,
                  'Quicklogger Git source blob must match current frozen source: ' + path)
    descriptor = h.binding(plan_path)
    return {'schema': 'megafmm.iter43.sw128-simt-control-measured-identity.v1', 'base_checkpoint': initial['base_checkpoint'],
        'measurement_source_commit': head, 'frozen_sources': initial['frozen_sources'],
        'frozen_binaries': initial['frozen_binaries'], 'predeclared_plan': descriptor,
        'predeclared_plan_sha256': descriptor['sha256'], 'quick_archive': str(archive),
        'independent_compiled_gate': gate, 'route_configurations': plan['route_configurations'],
        'linked_identity_path': initial['linked_identity_path'], 'linked_identity_sha256': initial['linked_identity_sha256'],
        'initial_identity': h.binding(initial_path), 'new_variants':plan['new_variants'],
        'new_source_paths':plan['new_source_paths'], 'comparison_pairs':plan['comparison_pairs'],
        'compiled_scope': 'Fresh original iteration43 compiled gate authorizes only its enabled instance scope. Source7/bin2 and linked identity are exact; initial pending history is preserved, never promoted to compiled proof. Numerical/performance certification remains separate.',
        'metadata_scope': 'Created ONCE after owner AKO quicklogger/Git closure and before mandatory full timing. Gate/link/planSHA/quickarchive/source commit and all source7/bin2 bindings are present initially; no hindsight append.'}


def exclusions(identity):
    rows = [{'algorithm': 'mega', 'variant': str(v), 'dtype': dtype, 'mode': mode,
        'classification': 'compiled_safety_blocked',
        'reason': 'Fresh current iteration43 compiled gate excludes this entire new variant; no timing substituted.',
        'evidence_path': identity['independent_compiled_gate']['path'],
        'evidence_sha256': identity['independent_compiled_gate']['sha256']}
        for v in (47,) if not identity['route_configurations'][str(v)]['enabled']
        for dtype in ('bf16', 'fp16') for mode in ('launch', 'graph')]
    if not rows:
        return None
    return {'schema': 'megafmm.iter43.measurement-exclusions.v1',
        **{key: identity[key] for key in ('measurement_source_commit', 'frozen_sources', 'frozen_binaries', 'predeclared_plan')},
        'unmeasured_rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('plan', 'gate', 'initial-identity', 'route-configurations', 'quick-archive'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', help='Owner-selected one-time measured identity; default stdout')
    parser.add_argument('--exclusions-output', help='Owner-selected exact4keys per blocked new variant')
    args = parser.parse_args(); h = helpers()
    value = generate(args.plan, args.gate, args.initial_identity, args.route_configurations, args.quick_archive)
    missing = exclusions(value)
    h.require(not args.exclusions_output or missing is not None, 'No exclusions file for full enabled scope')
    h.require(not (args.output and missing is not None) or args.exclusions_output,
              'A formal blocked identity requires its exact exclusions output in the same owner invocation')
    for path in (args.output, args.exclusions_output):
        h.require(not path or not Path(path).exists(), 'One-time artifact already exists: ' + str(path))
    h.emit(value, args.output)
    if args.exclusions_output:
        h.emit(missing, args.exclusions_output)
