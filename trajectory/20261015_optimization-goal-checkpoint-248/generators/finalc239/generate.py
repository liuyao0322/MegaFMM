from pathlib import Path
import subprocess
s=Path('results/opt_goal/static218/generate.py').read_text()
s=s.replace("s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)","s=subprocess.check_output(['git','show','1f1fb61:solution/sw_pipeline.cu'],text=True)\ns=s.replace('Threads=768','Threads=672').replace('pack_ready,608','pack_ready,512').replace('local+=2*608','local+=2*512')")
s=s.replace('int tid,int m0,int n0,float* D){','int tid){')
s=s.replace(' float* out=D+(n0+mn/64)*1024+m0+mn%64;',' uint32_t out=uint32_t(product_index(mn%64,mn/64))*4;')
s=s.replace('.reg .b64 ba,bb,da,db,dst;','.reg .b64 ba,bb,da,db; .reg .b32 dst,ob;')
s=s.replace("lines+=['add.u64 dst,%%5,%%%d;'%(38+j)]", "lines+=['xor.b32 ob,%%5,%%%d;'%(38+j),'add.u32 dst,%%56,ob;']")
s=s.replace("if base:lines+=['add.u64 dst,dst,%d;'%base]", "if base:lines+=['add.u32 dst,dst,%d;'%(dict([(2099200,3),(2048,1),(2097152,2)])[base]*8192)]")
s=s.replace("lines+=['st.global.f32 [dst],x;']", "lines+=['st.shared.f32 [dst],x;']\n   lines+=['bar.warp.sync 0xffffffff;','add.u32 rb,%%57,%d;'%(dict([(0,0),(2099200,3),(2048,1),(2097152,2)])[base]*8),'@leader mbarrier.arrive.shared::cta.b64 _,[rb];']")
s=s.replace("'\"l\"(out)'", "'\"r\"(out)'")
s=s.replace("ins+=['\"n\"(4*(%s%%64+(%s/64)*1024))'%(mn,mn)]", "ins+=['\"n\"(4*(%s/64*64+(%s%%64^((%s/64&6)<<2))))'%(mn,mn,mn)]")
s=s.replace("ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))']", "ins+=['\"n\"(int32_t(GMMA::Major::MN))','\"n\"(int32_t(GMMA::Major::K))','\"r\"(uint32_t(__cvta_generic_to_shared(s.output)))','\"r\"(uint32_t(__cvta_generic_to_shared(s.output_ready)))']")
s=s.replace('tensor_chain(s,tid,m0,n0,D);','tensor_chain(s,tid);')
s=s.replace("s=s[:a]+'  uint64_t ready[Stages],free[Stages];\\n'+s[b:]", "s=s[:a]+'  float output[4][Elements];\\n  uint64_t ready[Stages],free[Stages],output_ready[4];\\n'+s[b:]")
s=s.replace("s=s.replace('==213120','==196736')", "s=s.replace('==213120','==229504')")
s=s.replace("Path('solution/sw_pipeline.cu').write_text(s)",'''s=s.replace('    Barrier::init(&s.pack_ready,512);','    Barrier::init(&s.pack_ready,512);\\n    for(int j=0;j<4;++j)Barrier::init(&s.output_ready[j],4);')
s=s.replace('    arrive(&s.pack_ready);','''+"'''"+'''    arrive(&s.pack_ready);
    #pragma unroll
    for(int step=0;step<4;++step){
      int quadrant=step==0?0:step==1?3:step==2?1:2;
      wait(&s.output_ready[quadrant],0);
      unsigned row=unsigned(worker)/16,col=(unsigned(worker)%16)*4;
      unsigned shared_index=row*64+(col^((row&6)<<2));
      float4 value=*reinterpret_cast<const float4*>(s.output[quadrant]+shared_index);
      unsigned at=(n0+row+(quadrant/2)*512)*1024+m0+col+(quadrant%2)*512;
      *reinterpret_cast<float4*>(D+at)=value;
    }'''+"'''"+''')
Path('solution/sw_pipeline.cu').write_text(s)''')
s=s.replace("Path('solution/plan.cpp').write_text(s)","s=s.replace('smem=196736','smem=229504').replace('4wgmma+19pack_only','4wgmma+16pack_output').replace('product_layout=registers','product_layout=registers;final_C=shared_four_quadrants_xor;output_copy=float4_pack512')\nPath('solution/plan.cpp').write_text(s)")
exec(compile(s,'finalc-generator','exec'))
Path('solution/internal.h').write_text(subprocess.check_output(['git','show','62142fc:solution/internal.h'],text=True))
# Verify the XOR decomposition used by the inline PTX for every fragment element.
def index(m):return (m//64)*64+((m%64)^(((m//64)&6)<<2))
for lane in range(128):
 for j in range(16):
  lm=(lane%4)*128+((lane//4)%8)+(lane//32)*16
  jm=(j%2)*64+((j//2)%2)*8+(j//4)*512
  assert index(lm+jm)==index(lm)^index(jm)
