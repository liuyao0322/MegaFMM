from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
a=s.index('    for(int step=0;step<7;++step)for(int q=0;q<512/BK;++q){')
b=s.index('  }else if(tid>=128',a)
body=s[a:b];body=body[body.index('      // Raw P1/P2'):body.rindex('\n    }')]
out=''
for seq in range(14):
 step=seq//2;q=seq%2;p=[0,1,5,6,4,2,3][step]
 out+='    { constexpr int sequence=%d,slot=%d,k0=%d,p=%d;\n'%(seq,seq%4,q*256,p)+body+'\n    }\n'
s=s[:a]+out+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('operand_release=warp_leader','operand_release=warp_leader;tma_producer=full_static')
Path('solution/plan.cpp').write_text(s)
