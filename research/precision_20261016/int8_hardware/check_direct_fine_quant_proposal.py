"""CPU geometry/count checks for P55 proposal only; no C++ generation or GPU."""
from pathlib import Path
import hashlib
import json
import numpy as np

h = Path(__file__).resolve().parent
repo = h.parents[2]
for name in ['pack.cuh', 'corrected.cuh', 'compare.cu', 'pow2.cuh']:
    assert (h/name).read_bytes() == (h/'p51'/name).read_bytes()
indices = np.arange(768)[None, :]+768*np.arange(6)[:, None]
indices = indices[indices < 4096]
assert np.array_equal(np.sort(indices), np.arange(4096))
coverage_a = np.zeros((512, 512), np.uint8)
coverage_b = np.zeros((512, 512), np.uint8)
for task in range(64):
    coverage_a[8*task+indices//512, indices % 512] += 1
    coverage_b[indices//8, 8*task+indices % 8] += 1
assert np.all(coverage_a == 1) and np.all(coverage_b == 1)
metadata = 0
for n0 in range(0, 512, 32):
    for m0 in range(0, 512, 64):
        mn = np.arange(2048)
        a_local, b_local = (mn//64)//8, (mn % 64)//8
        assert np.all((n0//8+a_local) == (n0+mn//64)//8)
        assert np.all((m0//8+b_local) == (m0+mn % 64)//8)
        assert int(a_local.max()) == 3 and int(b_local.max()) == 7
        metadata += len(mn)
qa = lambda term: 7+term if term < 2 else (term-2)//2+(0 if term & 1 else 4)
qb = lambda term: 8-term if term < 2 else (term-2)//2+(4 if term & 1 else 0)
for t in [2, 4, 6]:
    assert (qa(t), qa(t+1), qb(t), qb(t+1)) == (4+(t-2)//2, (t-2)//2, (t-2)//2, 4+(t-2)//2)
native = 7*4*24576
integer = 2*4*12288+3*4*24576
commands = 7*4*2+2*4*2+3*4*4
work = 2*7*512**2*2+2*9*512**2+(9*64*2+1024)*4+262144
pack = ((5*24*4+5*4+9*512*12+127)//128)*128
consumer = ((8*24576+2*8*8+108*4+127)//128)*128
assert work == 12329472 and pack == 55808 and consumer == 197248
assert native+integer == 1081344 and commands == 120
assert 7*4*8+8*4*4 == 352
out = dict(candidate='P55', gpu_executed=False, cpp_generated=False,
    proposal_only=True, retained='P51', producer_grid=128, producer_threads=768,
    logical_packing_elements_per_side=int(coverage_a.size), producer_positions_per_task=4096,
    output_metadata_coordinates=metadata, metadata_elements_per_side=9*64,
    proposed_pack_shared_bytes=pack, proposed_consumer_shared_bytes=consumer,
    workspace_bytes=work, run_kernels=2, ordinary_descriptors=4,
    source_mma_instructions_per_cta=352, source_tma_bytes_per_cta=native+integer,
    source_tma_bytes_whole_grid=128*(native+integer), source_tma_commands_whole_grid=128*commands,
    actual_capacity_checked=False, hardware_accepted=False, performance_measured=False,
    integer_bound=2*512*127**2, reference='original rounded input FP64 AB, unchanged gates/all7+8 dots')
(h/'p55_proposal_cpu.json').write_text(json.dumps(out, indent=2)+'\n')
files = ['DIRECT_FINE_QUANT_DESIGN.md', 'check_direct_fine_quant_proposal.py',
         'p55_proposal_cpu.json', 'm15_manifest.json', 'm16_manifest.json',
         'p54_hardware_manifest.json', 'p39/pack.cuh', 'p39/corrected.cuh',
         'p39/compare.cu', 'p51/pack.cuh', 'p51/pow2.cuh']
(h/'p55_proposal_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):
    hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files}, indent=2)+'\n')
print(json.dumps(out, indent=2))
