"""Future P48 control: change only cooperative native scratch request to288B."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p47'
s=(base/'pack.cuh').read_text()
old='constexpr size_t smem=sizeof(PairPackShared<true>);'
assert s.count(old)==1
s=s.replace(old,'constexpr size_t smem=72*sizeof(float); //64 A partial maxima +8 B maxima')
old='args,sizeof(PairPackShared<true>),stream),"cooperative preprocess launch"'
assert s.count(old)==1
s=s.replace(old,'args,72*sizeof(float),stream),"cooperative preprocess launch"')
(h/'pack.cuh').write_text(s)
for name in ('corrected.cuh','compare.cu','pow2.cuh'):(h/name).write_bytes((base/name).read_bytes())
print('P48 generated: only cooperative attribute/occupancy-query/launch request65536→288bytes; P47 device code unchanged')
