#!/usr/bin/env python3
"""Owner-only CPU assembler preserving fresh compiled instances exactly."""
from pathlib import Path
import hashlib, json, re

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

gate_path = 'results/psplit_tma_bn16_k32_window_static_review_initial_iter46.json'
gate = json.loads(Path(gate_path).read_text())
compiled = {'compiled': gate['instances'], 'compiler_audit': {
    'original_review_path': gate_path, 'original_review_sha256': sha(gate_path),
    'status': gate['status'], 'compiled_gate_inherited_from_iteration45': False,
    'compiled_gate_inherited_from_parent49_or_any_prior': False,
    'scope': 'All original fresh50 fields preserved without aliases. Two ascending K32 input windows with A24/32 and wholeP24/32. Legal initial-owned preINC zeros and actual grown R80..91 definitions are recorded by the original fresh gate; no numerical/performance/residency certification.'}}
Path('results/psplit_tma_bn16_k32_window_compiled_metadata_iter46.json').write_text(json.dumps(compiled, indent=2) + '\n')

p = Path('results/small_psplit_tma_bn16_k32_window_environment_iter46.txt')
text = p.read_text()
assert int(p.with_suffix('.exit').read_text()) == 0
gpu = next(line for line in text.splitlines() if line.startswith('NVIDIA H100'))
fields = [f.strip() for f in gpu.split(',')]
version = re.search(r'^Version ([\d.]+) \(build (\d+)\)', text, re.M)
assert version
sm = re.search(r'DRIVER_API_MULTIPROCESSOR_COUNT (\d+)', text)
cutlass = re.findall(r'^f7b19de32c5d1f3cedfc735c2849f12b537522ee$', text, re.M)
assert sm and len(cutlass) == 1 and 'V12.4.99' in text
probe_path = 'results/psplit_tma_bn16_k32_window_function_occupancy_probe_iter46.txt'
probe_text = Path(probe_path).read_text()
probe, size = json.JSONDecoder().raw_decode(probe_text[probe_text.index('{'):])
assert not probe_text[probe_text.index('{') + size:].strip()
assert int(Path(probe_path).with_suffix('.exit').read_text()) == 0
meta = {'path': str(p), 'exit_path': str(p.with_suffix('.exit')), 'sha256': sha(p),
    'exit': 0, 'allocation': 107488, 'host': re.search(r'^HOST (\S+)', text, re.M)[1],
    'GPU': fields[0], 'gpu_uuid': fields[1], 'driver': fields[2], 'SMs': int(sm[1]),
    'SMs_evidence': 'Actual archived DriverAPI cuDeviceGetAttribute',
    'application_cuda': '12.4.99', 'CUTLASS': '3.5.1 commit ' + cutlass[0],
    'sanitizer_path': '/share/app/cuda/cuda-12.8/compute-sanitizer/compute-sanitizer',
    'sanitizer_version': version[1] + ' build' + version[2], 'clocks_locked': False,
    'snapshot_scope': 'Actual readonly postvalidation snapshot; no kernel launch or within-group clock stability/residency proof.',
    'function_attributes_and_occupancy_API': {'path': probe_path, 'sha256': sha(probe_path),
        'exit': 0, 'records': probe['records']}}
Path('results/psplit_tma_bn16_k32_window_environment_metadata_iter46.json').write_text(json.dumps(meta, indent=2) + '\n')
print('COMPILED_ENVIRONMENT_METADATA46_CLOSED')
