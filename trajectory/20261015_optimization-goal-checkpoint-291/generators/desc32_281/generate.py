from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','18f9a6f:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T> __device__ __forceinline__ void tensor_chain');b=s.index('template<class T> __global__ __launch_bounds__(512',a)
v=s[a:b]
v=v.replace('.reg .b64 ba,bb,da,db,dst;','.reg .b64 da,db,dst; .reg .b32 alo,ahi,blo,bhi,bal,bbl,dal,dbl;')
v=v.replace('  "setp.ne.u32 p,1,0;\\n"','  "mov.b64 {alo,ahi},%0; mov.b64 {blo,bhi},%1;\\n"\n  "setp.ne.u32 p,1,0;\\n"')
v=v.replace('add.u64 ba,%0,','add.u32 bal,alo,').replace('add.u64 bb,%1,','add.u32 bbl,blo,')
v=re.sub(r'add.u64 da,ba,(%\d+);',r'add.u32 dal,bal,\1; mov.b64 da,{dal,ahi};',v)
v=re.sub(r'add.u64 db,bb,(%\d+);',r'add.u32 dbl,bbl,\1; mov.b64 db,{dbl,bhi};',v)
assert 'add.u64 da' not in v and 'add.u64 ba' not in v
s=s[:a]+v+s[b:]
s=s.replace('// Zero-based P IDs.','// Descriptor address additions are confined to the low32 bits: all operand\n// addresses are below256KiB, so the14-bit start field cannot carry into metadata.\n// Zero-based P IDs.')
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','18f9a6f:solution/plan.cpp'],text=True).replace('recombine=private_static_ptx','recombine=private_static_ptx;descriptor_arithmetic=low32_fixed_high')
Path('solution/plan.cpp').write_text(p)
