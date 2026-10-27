from pathlib import Path
import subprocess
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace('for j in range(16)','for j in range(8)').replace('d<16>,first<16>,second<16>','d<8>,first<8>,second<8>').replace('m64n32k16','m64n16k16').replace('CLayout_64x32','CLayout_64x16').replace('%%54,%%55','%%46,%%47')
exec(compile(s,'dual243-generator','exec'))
p=Path('solution/sw_pipeline.cu');s=p.read_text()
s=s.replace('Shape<_32,_16>','Shape<_16,_16>')
s=s.replace('CLayout_64x16{}(tid-128,0)','CLayout_64x16{}((tid-128)%128,0)').replace('float* out=D+(n0+mn/64)','float* out=D+(n0+((tid-128)/128)*16+mn/64)').replace('uint64_t(db_tiles(0,0,0))','uint64_t(db_tiles(0,(tid-128)/128,0))')
s=s.replace('s.free[j],4','s.free[j],8').replace('pack_ready,608','pack_ready,480').replace('local+=2*608','local+=2*480').replace('|| tid>=256','|| tid>=384').replace('tid>=256 ? tid-160','tid>=384 ? tid-288').replace('tid<256','tid<384')
p.write_text(s)
p=Path('solution/plan.cpp');s=p.read_text().replace('4wgmma+19pack_only','8wgmma+15pack_only').replace('bk=256','bk=256;mma=64x16;consumer_groups=2;shared_lhs=1');p.write_text(s)
