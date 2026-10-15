"""P37 isolate identity-tile skip; preserve P36 doubleRN and vector processing."""
from pathlib import Path
h=Path(__file__).resolve().parent;base=h/'p36'
s=(base/'pack.cuh').read_text()
old=' int common=scratch[4];\n // With word stride256'
new=''' int common=scratch[4];
 // CTAuniform after the snapshot barrier. Original writers already fenced
 // before grid.sync; this path makes no new global writes or later barriers.
 if(scratch[0]==common&&scratch[1]==common&&scratch[2]==common&&scratch[3]==common)return;
 // With word stride256'''
assert s.count(old)==1;s=s.replace(old,new)
(h/'pack.cuh').write_text(s)
for name in ['corrected.cuh','compare.cu','pow2.cuh']:(h/name).write_text((base/name).read_text())
print('generated P37 exact skip for tiles whose four quantizer exponents already agree')
