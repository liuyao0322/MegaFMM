from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','492ba86:solution/sw_pipeline.cu'],text=True)
s=s.replace('counters+2*blockIdx.x+blockIdx.y/4','counters+32*(2*blockIdx.x+blockIdx.y/4)').replace('counters+32+2*blockIdx.y+blockIdx.x/8','counters+32*(32+2*blockIdx.y+blockIdx.x/8)').replace('counters+2*blockIdx.x+q','counters+32*(2*blockIdx.x+q)').replace('counters+32+2*blockIdx.y+q','counters+32*(32+2*blockIdx.y+q)')
s=s.replace('counters[2*blockIdx.x+blockIdx.y/4]','counters[32*(2*blockIdx.x+blockIdx.y/4)]').replace('counters[32+2*blockIdx.y+blockIdx.x/8]','counters[32*(32+2*blockIdx.y+blockIdx.x/8)]')
a='''        asm volatile("red.release.gpu.global.add.u32 [%0],1;" :: "l"(sf) : "memory");
        asm volatile("red.release.gpu.global.add.u32 [%0],1;" :: "l"(tf) : "memory");'''
b='''        // One GPU release fence orders transformed stores before both relaxed
        // RMW publications; acquire observations retain each release pattern.
        asm volatile("fence.acq_rel.gpu; red.relaxed.gpu.global.add.u32 [%0],1; red.relaxed.gpu.global.add.u32 [%1],1;" :: "l"(sf),"l"(tf) : "memory");'''
assert a in s;s=s.replace(a,b)
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','492ba86:solution/plan.cpp'],text=True).replace('impl->bytes=(size_t(4)<<20)+256;','impl->bytes=(size_t(4)<<20)+6144;').replace('workspace=4194560','workspace=4200448').replace('publication=early_coordinator_init_then_scoped_ST_counter','publication=early_init_scoped_ST_padded128B_single_release_fence')
Path('solution/plan.cpp').write_text(s)
