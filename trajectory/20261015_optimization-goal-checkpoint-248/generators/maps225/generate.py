from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','c0df342:solution/sw_pipeline.cu'],text=True)
s=s.replace('const T* A,const T* B,T* packed,float* D){','const T* A,const T* B,T* packed,float* D,const CUtensorMap* maps,bool bound_ab){')
s=s.replace('copy(&amap,','copy(maps && bound_ab ? maps : &amap,').replace('copy(&bmap,','copy(maps && bound_ab ? maps+1 : &bmap,').replace('copy(&smap,','copy(maps ? maps+2 : &smap,').replace('copy(&tmap,','copy(maps ? maps+3 : &tmap,')
s=s.replace('struct SWState{DType dtype;};','''struct SWState{
 DType dtype;CUtensorMap* maps=nullptr;const void* A=nullptr;const void* B=nullptr;void* workspace=nullptr;bool initialized=false;
 explicit SWState(DType t):dtype(t){check(cudaMalloc(reinterpret_cast<void**>(&maps),4*sizeof(CUtensorMap)),"SW immutable metadata");}
 ~SWState(){cudaFree(maps);}
};
void tune_sw_pipeline(SWState& state,const void* A,const void* B,void* workspace,cudaStream_t stream){
 if(state.initialized)return; // Existing captured graphs retain immutable descriptor contents.
 CUtensorMap maps[4]{};encode(maps[0],A,state.dtype,true,false);encode(maps[1],B,state.dtype,false,false);
 encode(maps[2],workspace,state.dtype,true,true);encode(maps[3],static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);
 check(cudaMemcpyAsync(state.maps,maps,sizeof(maps),cudaMemcpyHostToDevice,stream),"SW upload metadata");
 check(cudaStreamSynchronize(stream),"SW finish explicit metadata tuning");
 state.A=A;state.B=B;state.workspace=workspace;state.initialized=true;
}''').replace('std::make_shared<SWState>(SWState{p.dtype})','std::make_shared<SWState>(p.dtype)')
s=s.replace('  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D};dim3 grid(16,8);','''  const CUtensorMap* maps=state.initialized && workspace==state.workspace ? state.maps : nullptr;
  bool bound_ab=maps && A==state.A && B==state.B;
  void* args[]={&a,&b,&s,&t,&A,&B,&workspace,&D,&maps,&bound_ab};dim3 grid(16,8);''')
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','c0df342:solution/internal.h'],text=True)
s=s.replace('void run_sw_pipeline','void tune_sw_pipeline(SWState& state,const void* A,const void* B,void* workspace,cudaStream_t stream);\nvoid run_sw_pipeline')
Path('solution/internal.h').write_text(s)
s=subprocess.check_output(['git','show','c0df342:solution/plan.cpp'],text=True)
s=s.replace('    if (!impl.scale && !impl.sw)','    if(impl.sw)detail::tune_sw_pipeline(*impl.sw,A,B,workspace,stream);\n    if (!impl.scale && !impl.sw)')
s=s.replace('current_pointer_maps=4;','current_pointer_maps=4;immutable_metadata_bytes=512;bound_maps=tune_once_with_pointer_fallback;')
Path('solution/plan.cpp').write_text(s)
