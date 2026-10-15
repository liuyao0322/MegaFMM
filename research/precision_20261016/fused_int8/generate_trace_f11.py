"""Extend F9's role trace to frozen F11, with eight per-pack subphase stamps."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'generate_trace.py').read_text().replace("r/'f9/prototype.cuh'","r/'f11/prototype.cuh'").replace("r/'traced_f9.cuh'","r/'traced_f11.cuh'")
exec(compile(base,str(r/'generate_trace.py'),'exec'))
p=r/'traced_f11.cuh';s=p.read_text().replace('TraceCTA=272','TraceCTA=432')
s=s.replace('__device__ __forceinline__ void leaf_stamp','''__device__ __forceinline__ void pack_stamp(uint64_t* trace,int seq,int point){
 if(trace && threadIdx.x==0){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  trace[(blockIdx.y*gridDim.x+blockIdx.x)*TraceCTA+272+seq*8+point]=t;}
}
__device__ __forceinline__ void leaf_stamp''',1)
s=s.replace('int tid,float a0,float a1,float b0,float b1){','int tid,float a0,float a1,float b0,float b1,uint64_t* trace,int seq){\n pack_stamp(trace,seq,0);',1)
a=s.index('template<class T,int Prod,bool Fast>');b=s.index('// Both sides use the same two exponents',a)
body=s[a:b]
body=body.replace(' for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;\n  float v=', ' pack_stamp(trace,seq,1);\n for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;\n  float v=',1)
body=body.replace(' reduce4(s,peak,tid);',' pack_stamp(trace,seq,2);reduce4(s,peak,tid);pack_stamp(trace,seq,3);',1)
body=body.replace(' auto qbh=', ' pack_stamp(trace,seq,4);\n auto qbh=',1)
body=body.replace(' for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;float h=', ' pack_stamp(trace,seq,5);\n for(int i=tid;i<8192;i+=128){int n=i%64,k=i/64;float h=',1)
body=body.replace(' };\n if(eah>=', ' pack_stamp(trace,seq,6);\n };\n if(eah>=',1)
body=body.replace('"memory");producer_sync();','"memory");producer_sync();pack_stamp(trace,seq,7);',1)
s=s[:a]+body+s[b:]
s=s.replace('void pack_leaf(Shared& s,Slot& slot,int tid){','void pack_leaf(Shared& s,Slot& slot,int tid,uint64_t* trace,int seq){')
s=s.replace('s,slot,tid,a0,a1,b0,b1);','s,slot,tid,a0,a1,b0,b1,trace,seq);').replace('s,slot,tid,0,0,0,0);','s,slot,tid,0,0,0,0,trace,seq);')
s=s.replace('(s,s.slot[slot],tid);break;','(s,s.slot[slot],tid,trace,seq);break;')
for point in range(8):assert s.count(f'pack_stamp(trace,seq,{point})')==1,point
p.write_text(s)
h=(r/'trace.cu').read_text().replace('"traced_f9.cuh"','"traced_f11.cuh"').replace('t3,t4,t5\\n','t3,t4,t5,t6,t7\\n').replace("out<<'\\n';}","out<<\",0,0\\n\";}").replace('out<<",0,0\\n";','out<<",0,0,0,0\\n";')
# Leaf rows need exactly two pads; raw rows need four.
h=h.replace("for(int j=0;j<6;++j)out<<','<<ptr[seq*6+j];out<<\",0,0,0,0\\n\";","for(int j=0;j<6;++j)out<<','<<ptr[seq*6+j];out<<\",0,0\\n\";")
needle=' check(cudaEventDestroy(begin));'
h=h.replace(needle,''' for(int cta=0;cta<128;++cta)for(int seq=0;seq<20;++seq){out<<cta<<",0,pack,"<<seq;
  for(int j=0;j<8;++j)out<<','<<host[cta*fused_int8::TraceCTA+272+seq*8+j];out<<'\\n';}
'''+needle)
(r/'trace_f11.cu').write_text(h)
sh=(r/'build_trace.sh').read_text().replace('"$r/trace.cu"','"$r/trace_f11.cu"').replace('"$r/bin/trace"','"$r/bin/trace_f11"');(r/'build_trace_f11.sh').write_text(sh)
