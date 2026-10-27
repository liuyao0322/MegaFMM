from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','1bdfb29:solution/sw_pipeline.cu'],text=True)
s=s.replace('Barrier::init(&s.ready[j],1)','Barrier::init(&s.ready[j],2)')
a=s.index('  if(tid==0){\n    for(int step=0;step<7;');b=s.index('  }else if(tid>=128',a)
v=s[a:b].replace('  if(tid==0){','  if(tid==0 || tid==32){',1)
v=v.replace('TxBarrier::arrive_and_expect_tx(&s.ready[slot],2*(ASize+BSize));','TxBarrier::arrive_and_expect_tx(&s.ready[slot],tid==0?2*ASize:2*BSize);')
v=v.replace('      if(blo){','      if(tid==0){\n      if(blo){',1)
v=v.replace('      // RHS is A_leaf^T,','      }else{\n      // RHS is A_leaf^T,',1)
# Close the per-producer role branch before the loop body closes.
v=v[:v.rindex('\n    }')]+'\n      }'+v[v.rindex('\n    }'):]
s=s[:a]+v+s[b:]
Path('solution/sw_pipeline.cu').write_text(s)
p=subprocess.check_output(['git','show','1bdfb29:solution/plan.cpp'],text=True).replace('warp_roles=1tma+4wgmma','warp_roles=2tma+4wgmma;producer=LHS_warp0_RHS_warp1;ready_arrivals=2')
Path('solution/plan.cpp').write_text(p)
