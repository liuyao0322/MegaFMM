"""Freeze one completed standalone candidate and its exact local binary identity."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('label');args=ap.parse_args();label=args.label
assert re.fullmatch('f[0-9]+',label)
r=Path(__file__).resolve().parent;repo=r.parents[2];d=r/label;d.mkdir(exist_ok=False)
names=['prototype.cuh','compare.cu','build.sh',f'{label}_build.log',f'{label}_balanced.log',f'{label}_balanced.json']
for name in names:shutil.copy2(r/name,d/name)
binary=r/'bin'/f'compare_{label}';shutil.copy2(r/'bin/compare',binary)
paths=list(d.glob('*'))+[binary]
(r/f'{label}_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
