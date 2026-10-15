"""Summarize the two frozen P46 full profiler launches; never accept their timings."""
import csv
import hashlib
import json
from pathlib import Path

h = Path(__file__).resolve().parent
repo = h.parents[2]
raw = list(csv.reader((h/'ncu/profile_p46_raw.csv').open()))
assert len(raw) == 4
header, units = raw[:2]
fields = list(json.loads((h/'profile_p21_stats.json').read_text())['kernels'][0])
fields += [x for x in header if any(s in x for s in
           ('launch__occupancy_limit_', 'launch__waves_per_', 'launch__grid_size',
            'launch__block_size', 'launch__shared_mem_per_block_',
            'l1tex__data_bank_conflicts_', 'l1tex__data_pipe_lsu_wavefronts_'))]
fields = list(dict.fromkeys(fields))
indices = {key: header.index(key) for key in fields}
results = [{key: dict(value=row[i], unit=units[i]) for key, i in indices.items()}
           for row in raw[2:]]
for row, expected in zip(results, ['preprocess_cooperative', 'corrected']):
    assert expected in row['Kernel Name']['value']
out = dict(scope='first two balanced BF16 P46 full kernels, frozen full comparison --quick; '
                 'all profiler-process timing arms excluded from ordinary performance',
           kernels=results)
(h/'profile_p46_stats.json').write_text(json.dumps(out, indent=2)+'\n')
p = h/'profile_p46_details.log'
p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()).rstrip()+'\n')
files = ['profile_p46.sh', 'export_profile_p46.py', 'profile_p46_ncu.log',
         'profile_p46_details.log', 'profile_p46_stats.json', 'ncu/profile_p46.ncu-rep',
         'ncu/profile_p46_raw.csv', 'bin/compare_p46', 'p46_manifest.json',
         'profile_p21_stats.json']
(h/'profile_p46_manifest.json').write_text(json.dumps({
    str((h/f).relative_to(repo)): hashlib.sha256((h/f).read_bytes()).hexdigest()
    for f in files}, indent=2)+'\n')
for row in results:
    print(json.dumps({key: row[key] for key in [
        'Kernel Name', 'gpu__time_duration.sum', 'launch__registers_per_thread',
        'launch__shared_mem_per_block_dynamic', 'launch__occupancy_limit_registers',
        'launch__occupancy_limit_shared_mem', 'launch__waves_per_multiprocessor',
        'smsp__issue_active.avg.pct_of_peak_sustained_active',
        'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
        'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']}))
