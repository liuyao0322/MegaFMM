from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','62142fc:solution/sw_pipeline.cu'],text=True)
s=s.replace(' int mn=GMMA::CLayout_64x32{}(tid-128,0);',''' uint32_t lane=uint32_t(tid)&127u;
 uint32_t mn=((lane&3u)<<7)|((lane&28u)>>2)|((lane&96u)>>1);
 static_assert([]{for(int t=0;t<128;++t)if(GMMA::CLayout_64x32{}(t,0)!=int(((t&3)<<7)|((t&28)>>2)|((t&96)>>1)))return false;return true;}(),"fragment index");''')
s=s.replace('    int worker=','    unsigned worker=').replace('for(int local=2*worker;','for(unsigned local=2*worker;').replace('    int i=(blockIdx','    unsigned i=(blockIdx').replace('    int row=i/512,col=i%512,at=row*1024+col;','    unsigned row=i/512,col=i%512,at=row*1024+col;')
Path('solution/sw_pipeline.cu').write_text(s)
for name in ['internal.h','plan.cpp']:
 s=subprocess.check_output(['git','show','62142fc:solution/'+name],text=True)
 if name=='plan.cpp':s=s.replace('output=transposed_fragment','output=transposed_fragment;indexing=unsigned_pack_explicit_lane_bits')
 Path('solution/'+name).write_text(s)
