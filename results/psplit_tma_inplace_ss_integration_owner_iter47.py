from pathlib import Path
import json,hashlib
H=lambda b:hashlib.sha256(b).hexdigest(); ops=[]
def edit(path,old,new,count=1):
 p=Path(path);s=p.read_text();assert s.count(old)==count,(path,old[:60],s.count(old),count)
 before=H(s.encode());s=s.replace(old,new);p.write_text(s)
 ops.append({'path':path,'old':old,'new':new,'count':count,'before_sha256':before,'after_sha256':H(s.encode())})
edit('include/megafmm/kernels.h','// Native50 paired-K16 windows','''// Native51/52 inplace SS, current A/B TMA; columns32/16, configuration0.
bool wgmma_psplit_tma_inplace_ss_can_implement(const Problem& p,int columns,int configuration=0);
bool wgmma_psplit_tma_inplace_ss_full_tiles(const Problem& p,const void* A,const void* B,int columns,int configuration=0);
void prepare_wgmma_psplit_tma_inplace_ss(const Problem& p,int columns,int configuration=0);
void launch_wgmma_psplit_tma_inplace_ss(const Problem& p,const void* A,const void* B,
    const float* C0,float* D,cudaStream_t stream,int columns,int configuration=0);

// Native50 paired-K16 windows''')
edit('include/megafmm/gemm.h','  int mega_variant=0;','''  // 51/52=BN32/16 BK32 inplace final shared ST, eight currentAB TMA boxes.
  // Fresh four-instance resource/sync gate required; original precision floors.
  int mega_variant=0;''')
edit('bench/benchmark.cu','--mega-variant 0..50','--mega-variant 0..52')
edit('bench/benchmark.cu','o.mega_variant > 50','o.mega_variant > 52')
edit('bench/benchmark.cu','mega-variant must be between 0 and 50','mega-variant must be between 0 and 52')
edit('CMakeLists.txt','solution/megakernel_wgmma_psplit_tma_bn16_k32_window.cu)','solution/megakernel_wgmma_psplit_tma_bn16_k32_window.cu solution/megakernel_wgmma_psplit_tma_inplace_ss.cu)')
p=Path('CMakeLists.txt');p.write_text(p.read_text()+'''\nset_source_files_properties(solution/megakernel_wgmma_psplit_tma_inplace_ss.cu
  PROPERTIES COMPILE_OPTIONS "--keep;--keep-dir=${CMAKE_CURRENT_SOURCE_DIR}/.scratch/psplit_tma_inplace_ss_nvcc_keep_iter47")\n''')
ops.append({'path':'CMakeLists.txt','append':'set_source_files_properties new TU keep-dir iter47'})
edit('solution/plan.cpp','''    if (plan.policy.mega_variant == 50) {''','''    if (plan.policy.mega_variant == 51 || plan.policy.mega_variant == 52) {
      const int columns=plan.policy.mega_variant==51 ? 32 : 16;
      if (wgmma_psplit_tma_inplace_ss_full_tiles(p,A,B,columns,0))
        launch_wgmma_psplit_tma_inplace_ss(p,A,B,C0,D,stream,columns,0);
      else launch_packed_mma(p,A,B,C0,D,stream,7);
    } else if (plan.policy.mega_variant == 50) {''')
edit('solution/plan.cpp','''      if (impl_->policy.mega_variant == 50) {''','''      if (impl_->policy.mega_variant == 51 || impl_->policy.mega_variant == 52) {
        const int columns=impl_->policy.mega_variant==51 ? 32 : 16;
        if (wgmma_psplit_tma_inplace_ss_can_implement(impl_->problem,columns,0))
          return columns==32 ? "megakernel_wgmma_psplit_tma_inplace_ss_64x32" : "megakernel_wgmma_psplit_tma_inplace_ss_64x16";
        return "megakernel_warp_mma_packed";
      }
      if (impl_->policy.mega_variant == 50) {''')
edit('solution/plan.cpp','policy.mega_variant > 50','policy.mega_variant > 52')
edit('solution/plan.cpp','mega_variant must be between 0 and 50','mega_variant must be between 0 and 52')
edit('solution/plan.cpp','''    const bool supported = policy.mega_variant == 50''','''    const bool supported = policy.mega_variant == 51 || policy.mega_variant == 52
                             ? (wgmma_psplit_tma_inplace_ss_can_implement(p,policy.mega_variant==51 ? 32 : 16,0) || packed_mma_can_implement(p,7))
                           : policy.mega_variant == 50''')
edit('solution/plan.cpp','''    if (policy.mega_variant == 50) {''','''    if (policy.mega_variant == 51 || policy.mega_variant == 52) {
      const int columns=policy.mega_variant==51 ? 32 : 16;
      if (wgmma_psplit_tma_inplace_ss_can_implement(p,columns,0)) prepare_wgmma_psplit_tma_inplace_ss(p,columns,0);
      else prepare_packed_mma(p,7);
    } else if (policy.mega_variant == 50) {''')
Path('results/psplit_tma_inplace_ss_integration_delta_iter47.json').write_text(json.dumps({'schema':'megafmm.iter47.inplace-ss-integration-delta.v1','operations':ops,'scope':'only new template51/52 CMake/API/Plan/bench declarations; tests independentchild delta; source-only no runtime proof'},indent=2)+'\n')
# Nonexecuting first-source comment clarification before first compilation.
f=Path('solution/megakernel_wgmma_psplit_tma_inplace_ss.cu');s=f.read_text();dpath=Path('results/psplit_tma_inplace_ss_source_delta_iter47.json');d=json.loads(dpath.read_text())
for old,new in [('Shared layout construction uses existing SW32 atoms. Each bank has its','Shared layout uses SW64 A, SW64 BN32 B or SW32 BN16 B. Each bank has its'),('Complete all7168 FP32 P words','Complete all7*BM*BN FP32 P words')]:
 assert s.count(old)==1;before=H(s.encode());s=s.replace(old,new);d['steps'].append({'old':old,'new':new,'count':1,'before_sha256':before,'after_sha256':H(s.encode())})
f.write_text(s);d['new_sha256']=H(s.encode());dpath.write_text(json.dumps(d,indent=2)+'\n')
Path('.scratch/psplit_tma_inplace_ss_nvcc_keep_iter47').mkdir()
print(json.dumps({'source':d['new_sha256'],'source_delta_steps':len(d['steps']),'Plan':H(Path('solution/plan.cpp').read_bytes()),'tests':H(Path('tests/test_correctness.cu').read_bytes()),'integration_ops':len(ops)}))
