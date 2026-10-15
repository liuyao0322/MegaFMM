"""Freeze a completed research candidate; never edits production sources."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('label');args=ap.parse_args()
r=Path(__file__).resolve().parent;label=args.label
assert label.startswith('p') and label[1:].isdigit()
target=r/label;target.mkdir(exist_ok=False)
names=['pack.cuh','corrected.cuh','compare.cu','pow2.cuh',f'{label}_build.log',f'{label}_balanced.log',f'{label}_balanced.json']
for name in names:shutil.copy2(r/name,target/name)
binary=r/'bin'/f'compare_{label}';shutil.copy2(r/'bin/compare',binary)
paths=list(target.glob('*'))+[binary]
repo=r.parents[2]
manifest={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(r/f'{label}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
