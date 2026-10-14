from pathlib import Path
s=Path('results/opt_goal/n48pdl257/generate.py').read_text()
s=s.replace('BN=48','BN=40').replace('grid(11,8)','grid(13,8)').replace('dim3(11,8)','dim3(13,8)').replace('Count<88','Count<104')
s=s.replace('for j in range(24)','for j in range(20)').replace('d<24>,first<24>,second<24>','d<20>,first<20>,second<20>').replace('m64n48k16','m64n40k16')
s=s.replace('Shape<_48,','Shape<Int<40>,').replace('slot*3584','slot*3328').replace('%%62,%%63','%%58,%%59').replace('%%64','%%60').replace('229504','213120')
s=s.replace('tile=64x48;ctas=88','tile=64x40;ctas=104')
exec(compile(s,'n40-pdl-generator','exec'))
p=Path('solution/sw_pipeline.cu');s=p.read_text()
# CLayout64x48 has the same first20 per-thread values as64x40.
# Our PTX only uses those20 values; no dependency on a missing CuTe alias.
s=s.replace('  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','''  int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;
  if(tid==32){asm volatile("prefetch.tensormap [%0]; prefetch.tensormap [%1]; prefetch.tensormap [%2]; prefetch.tensormap [%3];" :: "l"(&amap),"l"(&bmap),"l"(&smap),"l"(&tmap) : "memory");}''')
assert 'BN=40' in s
assert 'm64n40k16' in s
p.write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('current_pointer_maps=4','current_pointer_maps=4;descriptor_prefetch=warp1_before_init_barrier');p.write_text(s)
