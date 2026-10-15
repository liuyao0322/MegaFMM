"""Freeze P9/P10 device paths for fresh-input diagnostics without unbalanced variants."""
from pathlib import Path
r=Path(__file__).resolve().parent
for label in ('p9','p10'):
    source=r/label;dest=r/('alternating_'+label);dest.mkdir(exist_ok=False)
    for name in ('pack.cuh','corrected.cuh','pow2.cuh'):
        s=(source/name).read_text()
        if name=='corrected.cuh':
            start=s.index('void run(megafmm::DType')
            s=s[:start]+'} // namespace int8_research\n'
        s=s.replace('namespace int8_research','namespace alternating_'+label)
        (dest/name).write_text(s)
