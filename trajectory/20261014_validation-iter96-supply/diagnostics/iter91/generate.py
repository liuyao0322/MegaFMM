from pathlib import Path
import subprocess,hashlib,json,sys
out=Path(sys.argv[1] if len(sys.argv)>1 else 'results/production_chain/source')
out.mkdir(parents=True,exist_ok=True)
s=subprocess.check_output(['git','show','3a527f9:solution/sw_pipeline.cu']).decode().split('struct SWState')[0]
s+='} // namespace megafmm::detail\n'
(out/'baseline.cuh').write_text(s.replace('namespace megafmm::detail','namespace megafmm::diagnostic_base'))
t=s.replace('namespace megafmm::detail','namespace megafmm::diagnostic_trace')
t=t.replace('template<class T> __global__ __launch_bounds__(Threads,1)','template<class T,int Role> __global__ __launch_bounds__(Threads,1)')
t=t.replace('const T* A,const T* B,T* packed,float* D){','const T* A,const T* B,T* packed,float* D,Trace* traces,int sample){')
t=t.replace('int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;','int tid=threadIdx.x,m0=blockIdx.y*BM,n0=blockIdx.x*BN;\n  uint64_t t0=0,t1=0,t2=0,t3=0;\n  if constexpr(Role==4)t0=stamp();')
t=t.replace('cooperative_groups::this_grid().sync();','cooperative_groups::this_grid().sync();\n  if constexpr(Role==4)t1=stamp();')
t=t.replace('if(sequence>=2)wait(&s.free[slot],((sequence-2)/2)%2);','if constexpr(Role==1)if(sequence==sample)t0=stamp();\n      if(sequence>=2)wait(&s.free[slot],((sequence-2)/2)%2);\n      if constexpr(Role==1)if(sequence==sample)t1=stamp();')
# TMA issue ends just before the producer loop closes.
a='''      }
    }
  }else if((tid>=32'''
b='''      }
      if constexpr(Role==1)if(sequence==sample){t2=stamp();t3=t2;}
    }
  }else if((tid>=32'''
assert a in t;t=t.replace(a,b)
t=t.replace('wait(&s.product_ready[step%2],(step/2)%2);','if constexpr(Role==2)if(step==sample)t0=stamp();\n      wait(&s.product_ready[step%2],(step/2)%2);\n      if constexpr(Role==2)if(step==sample)t1=stamp();')
t=t.replace('arrive(&s.product_free[step%2]);','if constexpr(Role==2)if(step==sample)t2=stamp();\n      arrive(&s.product_free[step%2]);')
a='''      }
    }
  }else if(tid>=128'''
b='''      }
      if constexpr(Role==2)if(step==sample)t3=stamp();
    }
  }else if(tid>=128'''
assert a in t;t=t.replace(a,b)
t=t.replace('wait(&s.ready[slot],(sequence/2)%2);','if constexpr(Role==0)if(sequence==sample)t0=stamp();\n        wait(&s.ready[slot],(sequence/2)%2);\n        if constexpr(Role==0)if(sequence==sample)t1=stamp();')
t=t.replace('warpgroup_commit_batch();warpgroup_wait<0>();','warpgroup_commit_batch();\n        if constexpr(Role==0)if(sequence==sample)t2=stamp();\n        warpgroup_wait<0>();\n        if constexpr(Role==0)if(sequence==sample)t3=stamp();')
t=t.replace('if(step>=2)wait(&s.product_free[step%2],((step-2)/2)%2);','if constexpr(Role==3)if(step==sample)t0=stamp();\n      if(step>=2)wait(&s.product_free[step%2],((step-2)/2)%2);\n      if constexpr(Role==3)if(step==sample)t1=stamp();')
t=t.replace('arrive(&s.product_ready[step%2]);','if constexpr(Role==3)if(step==sample)t2=stamp();\n      arrive(&s.product_ready[step%2]);\n      if constexpr(Role==3)if(step==sample)t3=stamp();')
a='''  __syncthreads();
}
void check'''
b='''  if constexpr(Role==4)t2=stamp();
  __syncthreads();
  if constexpr(Role==4)t3=stamp();
  if((Role==0 || Role==3 || Role==4) ? tid==128:Role==1 ? tid==0:tid==32)
    traces[blockIdx.y*gridDim.x+blockIdx.x]={t0,t1,t2,t3};
}
void check'''
assert a in t;t=t.replace(a,b)
t=t.replace('uint64_t t0=0,t1=0,t2=0,t3=0;', 'uint64_t t0=0,t1=0,t2=0,t3=0,phase_begin=0,production_begin=0; uint32_t ready_ticks=0,issue_ticks=0,drain_ticks=0;')
t=t.replace('  }else if(tid>=128 && tid<256){', '  }else if(tid>=128 && tid<256){\n    if constexpr(Role==5)production_begin=stamp();')
t=t.replace('if constexpr(Role==0)if(sequence==sample)t0=stamp();', 'if constexpr(Role==5)phase_begin=stamp();\n        if constexpr(Role==0)if(sequence==sample)t0=stamp();')
t=t.replace('if constexpr(Role==0)if(sequence==sample)t1=stamp();', 'if constexpr(Role==0)if(sequence==sample)t1=stamp();\n        if constexpr(Role==5){uint64_t end=stamp();ready_ticks+=uint32_t(end-phase_begin);phase_begin=end;}')
t=t.replace('if constexpr(Role==0)if(sequence==sample)t2=stamp();', 'if constexpr(Role==0)if(sequence==sample)t2=stamp();\n        if constexpr(Role==5){uint64_t end=stamp();issue_ticks+=uint32_t(end-phase_begin);phase_begin=end;}')
t=t.replace('if constexpr(Role==0)if(sequence==sample)t3=stamp();', 'if constexpr(Role==0)if(sequence==sample)t3=stamp();\n        if constexpr(Role==5){uint64_t end=stamp();drain_ticks+=uint32_t(end-phase_begin);}')
t=t.replace('    }\n  }\n  if constexpr(Role==4)t2=stamp();', '    }\n    if constexpr(Role==5){t0=ready_ticks;t1=issue_ticks;t2=drain_ticks;t3=stamp()-production_begin;}\n  }\n  if constexpr(Role==4)t2=stamp();')
t=t.replace('Role==3 || Role==4)', 'Role==3 || Role==4 || Role==5)')
(out/'instrumented.cuh').write_text(t)
(out/'metadata.json').write_text(json.dumps({'commit':'3a527f9','source_sha256':hashlib.sha256(subprocess.check_output(['git','show','3a527f9:solution/sw_pipeline.cu'])).hexdigest(),'clock':'%globaltimer nanoseconds','roles':{'0':'TC operand wait, issue, drain','1':'TMA slot wait and issue','2':'SIMT P wait, read, recombine','3':'TC P-slot wait, store, publish','4':'CTA transform+grid, production chain, final CTA barrier','5':'same-invocation TC cumulative nanoseconds: t0=operand_wait,t1=prepare_issue,t2=explicit_wait0,t3=producer_elapsed'}},indent=2)+'\n')

# Account for residual readiness separately: preserve prepare/issue versus waits.
p=out/'instrumented.cuh'
t=p.read_text()
a='if(p>=2)wait(&s.low_ready[slot],(sequence/2)%2);'
b='if(p>=2){\n          if constexpr(Role==5){uint64_t end=stamp();issue_ticks+=uint32_t(end-phase_begin);phase_begin=end;}\n          wait(&s.low_ready[slot],(sequence/2)%2);\n          if constexpr(Role==5){uint64_t end=stamp();ready_ticks+=uint32_t(end-phase_begin);phase_begin=end;}\n        }'
assert a in t
t=t.replace(a,b)
p.write_text(t)
