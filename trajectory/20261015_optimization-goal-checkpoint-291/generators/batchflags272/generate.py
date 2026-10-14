from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','50f361d:solution/sw_pipeline.cu'],text=True)
a=s.index('        auto await_tile=');b=s.index('\n      }\n      if(sequence>=Stages)',a)
lines=['{ .reg .b32 v<12>,all; .reg .pred ready;','POLL_TILES:']
for j in range(8):lines.append(f'ld.relaxed.gpu.global.u32 v{j},[%0+{j*1024}];')
for j in range(4):lines.append(f'ld.relaxed.gpu.global.u32 v{8+j},[%1+{j*128}];')
lines.append('and.b32 all,v0,v1;')
for j in range(2,12):lines.append(f'and.b32 all,all,v{j};')
lines+=['setp.ne.u32 ready,all,0;','@!ready bra POLL_TILES;','fence.acq_rel.gpu;','}']
body='''        // Issue independent relaxed observations together, then acquire every
        // producer release with a single GPU-scope acquire fence.
        auto lhs=flags+((k0/32)*8+m0/64)*32;
        auto rhs=flags+((n0/32)*8+k0/64)*32;
        asm volatile(\n'''+''.join('          "'+x+'\\n"\n' for x in lines)+'''          : : "l"(lhs),"l"(rhs) : "memory");'''
s=s[:a]+body+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','50f361d:solution/plan.cpp'],text=True).replace('tile_release_acquire','tile_release_batch_relaxed_then_acquire_fence')
Path('solution/plan.cpp').write_text(s)
