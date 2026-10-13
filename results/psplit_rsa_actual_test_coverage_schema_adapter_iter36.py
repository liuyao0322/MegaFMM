#!/usr/bin/env python3
"""CPU-only schema adapter; preserve the frozen iteration36 audit generator."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / '.scratch/psplit_rsa_actual_test_coverage_generator_iter36.py'
GENERATOR_SHA = 'a496d934b35c8e9dc87741afa4d046b5b527c2976e4ad9a933f6452a151adc51'
LINK = 'results/psplit_rsa_initial_link_identity_iter36.json'
LINK_SHA = '21c2e01916f1ac12a3af8074bb14c1670d9ecfc46c2bfc57645e435b24b08294'
GATE_SHA = '46db33836552063d31dd3cfde249ddb472cf0bdd252e2a9e4f790690362a1488'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--formal-expected', required=True)
    args = parser.parse_args()
    identity_path = ROOT / args.identity
    identity = json.loads(identity_path.read_text())
    descriptor = identity['independent_compiled_gate']
    gate_path = ROOT / descriptor['path']
    assert sha(gate_path) == descriptor['sha256'] == GATE_SHA
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == descriptor['status'] == 'PASS_COMPILED_CAN_LAUNCH_39_ONLY_37_REMAINS_BLOCKED'
    assert gate['frozen_bindings'][LINK] == sha(ROOT / LINK) == LINK_SHA
    link = json.loads((ROOT / LINK).read_text())
    for domain in ('frozen_sources', 'frozen_binaries'):
        for name, wanted in identity[domain].items():
            assert gate['frozen_bindings'][name] == wanted == sha(ROOT / name)
            if domain == 'frozen_binaries':
                assert link[name] == wanted
    addendum = identity['measurement_plan_addendum']
    assert sha(ROOT / addendum['path']) == addendum['sha256']
    assert identity['predeclared_plan_sha256'] == identity['predeclared_plan']['sha256']
    assert sha(GENERATOR) == GENERATOR_SHA
    original = GENERATOR.read_text()
    old = "    link_path = identity['linked_identity_path']"
    new = "    link_path = identity.get('linked_identity_path', 'results/psplit_rsa_initial_link_identity_iter36.json')"
    assert original.count(old) == 1
    adapted = original.replace(old, new)
    write_line = '    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + \'\\n\')'
    assert adapted.count(write_line) == 1
    insertion = "    result['CPU_schema_adapter_provenance'] = ADAPTER_PROVENANCE\n"
    adapted = adapted.replace(write_line, insertion + write_line)
    provenance = {
        'reason': 'Measured identity iteration36 omits the older linked_identity_path field; the current independent compiled gate binds the original linked-identity artifact.',
        'original_generator': {'path': str(GENERATOR.relative_to(ROOT)), 'sha256': GENERATOR_SHA, 'original_bytes_preserved': True},
        'archived_adapter': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha(__file__)},
        'in_memory_execution_source_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'exact_parser_schema_replacement': {'original': old, 'replacement': new},
        'exact_provenance_insertion_before_sole_output_write': insertion,
        'bound_linked_identity': {'path': LINK, 'sha256': LINK_SHA},
        'bound_compiled_gate': descriptor,
        'measured_identity_sha256': sha(identity_path),
        'gate_source7_binary2_maps_match_measured_identity_and_current_files': True,
        'both_linked_binary_hashes_match_measured_identity': True,
        'measurement_plan_addendum_verified': addendum,
        'formal_expected_measured_identity_and_generator_not_modified': True,
        'CPU_only_no_GPU_build_or_Git_mutation': True,
    }
    namespace = {'__file__': str(GENERATOR), '__name__': 'iteration36_cpu_schema_adapted_generator',
                 'ADAPTER_PROVENANCE': provenance}
    exec(compile(adapted, str(GENERATOR), 'exec'), namespace)
    namespace['run'](args)
    assert sha(GENERATOR) == GENERATOR_SHA


if __name__ == '__main__':
    main()
