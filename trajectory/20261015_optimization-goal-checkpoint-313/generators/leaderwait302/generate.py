from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
a=s.index('template<class T> __device__ __forceinline__ void tensor_chain');b=s.index('template<class T> __global__ __launch_bounds__(512',a);v=s[a:b]
pat=r'  "WAIT_(\d+)_(\d+):\\n"\n  "mbarrier.try_wait.parity.shared::cta.b64 done,\[rb\],([01]),0x989680;\\n"\n  "@!done bra WAIT_\1_\2;\\n"'
def rep(m):
 st,q,ph=m.groups();lab=f'WAIT_{st}_{q}';ready=f'READY_{st}_{q}'
 return '\n'.join('  "'+x+'\\n"' for x in [f'@!leader bra {ready};',lab+':',f'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],{ph},0x989680;',f'@!done bra {lab};',ready+':','bar.warp.sync 0xffffffff;'])
v,n=re.subn(pat,rep,v);assert n==56,n
s=s[:a]+v+s[b:];Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=warp_leader;ready_wait=warp_leader_then_sync')
Path('solution/plan.cpp').write_text(p)
