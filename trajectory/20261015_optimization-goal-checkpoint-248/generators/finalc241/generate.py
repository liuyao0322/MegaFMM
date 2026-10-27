from pathlib import Path
import subprocess,re
s=subprocess.check_output(['git','show','3e32a15:solution/sw_pipeline.cu'],text=True)
s=s.replace('float output[4][Elements];','float output[2][Elements];').replace('output_ready[4];','output_ready[4],output_free[2];').replace('==229504','==213120')
s=s.replace('for(int j=0;j<4;++j)Barrier::init(&s.output_ready[j],4);','for(int j=0;j<4;++j)Barrier::init(&s.output_ready[j],4);\n    for(int j=0;j<2;++j)Barrier::init(&s.output_free[j],16);')
s=re.sub(r'add.u32 dst,dst,(24576|8192|16384);',lambda m:'add.u32 dst,dst,%d;'%{24576:8192,8192:0,16384:8192}[int(m[1])],s)
s=s.replace('s.output[quadrant]+shared_index','s.output[step%2]+shared_index').replace('*reinterpret_cast<float4*>(D+at)=value;','*reinterpret_cast<float4*>(D+at)=value;\n      __syncwarp();if((tid&31)==0)arrive(&s.output_free[step%2]);')
s=s.replace('"r"(uint32_t(__cvta_generic_to_shared(s.output_ready))) : "memory"','"r"(uint32_t(__cvta_generic_to_shared(s.output_ready))), "r"(uint32_t(__cvta_generic_to_shared(s.output_free))) : "memory"')
for step,first,slot in [(5,'add.rn.f32 x,first0,d0;',0),(6,'sub.rn.f32 x,second0,d0;',1)]:
 start=0
 while True:
  a=s.find('  "WAIT_%d_1:'%step,start)
  if a<0:break
  b=s.index('  "'+first,a)
  lines=['add.u32 rb,%%58,%d;'%(slot*8),'WAIT_OUTPUT_%d:'%step,'mbarrier.try_wait.parity.shared::cta.b64 done,[rb],0,0x989680;','@!done bra WAIT_OUTPUT_%d;'%step]
  add=''.join('  "'+x+'\\n"\n' for x in lines)
  s=s[:b]+add+s[b:];start=b+len(add)+len(first)
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','3e32a15:solution/plan.cpp'],text=True).replace('smem=229504','smem=213120').replace('final_C=shared_four_quadrants_xor','final_C=shared_two_slots_xor;output_free=16_warp_leaders')
Path('solution/plan.cpp').write_text(s)
