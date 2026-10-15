"""P42: native-high exponent bit formula, independent of P40/P41 changes."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p39'
s=(base/'pack.cuh').read_text()
helper=(h/'quant_exponent_proposal.cuh').read_text().replace('#pragma once\n','')+'\n'
anchor='// Keep only two FP32 transformed planes live. All phases share one scratch area.'
assert s.count(anchor)==1;s=s.replace(anchor,helper+anchor)
start=s.index('template<class T,bool SideA,int NativePlane,int QP>')
for peak in ['peak','peak0','peak1']:
    old=f'{peak}>0?int(ceilf(log2f({peak}/127.f))):0'
    assert s[start:].count(old)==1
    s=s[:start]+s[start:].replace(old,f'native_high_exponent({peak})')
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P42 independent native-high exponent path, original tiny fallback')
