"""Regenerate independent baseline/candidate compilation inputs from commits."""
from pathlib import Path
import subprocess
root=Path('research/falcon_layout_fusion')
baseline=subprocess.check_output(['git','show','6d4167b:solution/sw_pipeline.cu'],text=True).replace('namespace megafmm::detail {','namespace megafmm::diagnostic_base {')
versions={332:'e4dd218',333:'28b4344',335:'35817d1'}
# Existing check.cu and multistream.cu are preserved alongside this script.
for n,sha in versions.items():
 d=root/f'check{n}';d.mkdir(exist_ok=True)
 (d/'baseline.cuh').write_text(baseline)
 source=subprocess.check_output(['git','show',f'{sha}:solution/sw_pipeline.cu'],text=True)
 (d/'candidate.cuh').write_text(source.replace('namespace megafmm::detail {','namespace megafmm::diagnostic_candidate {'))
