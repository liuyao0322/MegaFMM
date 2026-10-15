"""Extract one diagnostic NCU kernel's metrics with explicit exported units."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--candidate',required=True);args=ap.parse_args()
h=Path(__file__).resolve().parent;repo=h.parents[2];label=args.label
rows=list(csv.reader((h/'ncu'/f'{label}_raw.csv').open()));assert len(rows)==3
units=dict(zip(rows[0],rows[1]));values=dict(zip(rows[0],rows[2]))
keys=[k for k in values if 'bank_conflicts' in k or ('wavefronts' in k and 'shared' in k) or ('sectors' in k and 'local_op_' in k) or k in ['gpu__time_duration.sum','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_no_instruction_per_warp_active.pct','launch__registers_per_thread','launch__shared_mem_per_block_static']]
out=dict(scope=f'one BF16 frozen {args.candidate} kernel, diagnostic only; unlocked clocks, no cache flush',metrics={k:dict(value=values[k],unit=units[k]) for k in keys})
(h/f'{label}_stats.json').write_text(json.dumps(out,indent=2)+'\n')
f=h/f'{label}_details.log';f.write_text('\n'.join(s.rstrip() for s in f.read_text().splitlines()).rstrip()+'\n')
files=[f'{label}.cu',f'build_{label}.sh',f'{label}.sh',f'{label}_build.log',f'{label}_ncu.log',f'{label}_details.log',f'{label}_stats.json',f'ncu/{label}.ncu-rep',f'ncu/{label}_raw.csv',f'bin/{label}',f'{args.candidate}/pack.cuh',f'{args.candidate}/pow2.cuh','summarize_profile.py']
(h/f'{label}_manifest.json').write_text(json.dumps({str((h/f).relative_to(repo)):hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print(json.dumps(out,indent=2))
