from pathlib import Path
import subprocess
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)","s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)\ns=s.replace('BN=32','BN=64').replace('Stages=4','Stages=3').replace('Threads=768','Threads=512').replace('grid(16,8)','grid(8,8)').replace('Plane/(16*8)','Plane/(8*8)').replace('pack_ready,608','pack_ready,352').replace('local+=2*608','local+=2*352').replace('80000000u-127u','80000000u-63u').replace('Count<128','Count<64')")
s=s.replace('for j in range(16)', 'for j in range(32)')
s=s.replace('Shape<_32,','Shape<_64,').replace('CLayout_64x32','CLayout_64x64').replace('m64n32k16','m64n64k16').replace('d<16>,first<16>,second<16>','d<32>,first<32>,second<32>').replace('range(16)]','range(32)]').replace('for j in range(16):','for j in range(32):').replace('slot=seq%4;phase=seq//4%2','slot=seq%3;phase=seq//3%2').replace('slot*3072','slot*4096').replace('%%54,%%55','%%70,%%71')
s=s.replace("s=s.replace('==213120','==196736')","s=s.replace('==213120','==196736')")
s=s.replace("Path('solution/plan.cpp').write_text(s)","s=s.replace('4wgmma+19pack_only','4wgmma+11pack_only').replace('operand_stages=4','operand_stages=3').replace('bk=256','bk=256;tile=64x64;compute_ctas=64')\nPath('solution/plan.cpp').write_text(s)")
exec(compile(s,'n64-generator','exec'))
# Internal interface must also discard225's unretained metadata tuner.
Path('solution/internal.h').write_text(subprocess.check_output(['git','show','c0df342:solution/internal.h'],text=True))
