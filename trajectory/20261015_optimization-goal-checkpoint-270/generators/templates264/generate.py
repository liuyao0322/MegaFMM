from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','13340d1:solution/sw_pipeline.cu'],text=True)
s=s.replace('struct SWState{DType dtype;};','struct SWState{DType dtype;CUtensorMap maps[4];};')
s=s.replace('  return std::make_shared<SWState>(SWState{p.dtype});','''  auto state=std::make_shared<SWState>();state->dtype=p.dtype;
  // Encode invariant tensor geometry once. Every invocation replaces all four
  // addresses in private host copies, so plans remain safe across streams.
  void* placeholder=nullptr;
  check(cudaMalloc(&placeholder,16),"SW tensor template placeholder");
  try{
    encode(state->maps[0],placeholder,p.dtype,true,false);
    encode(state->maps[1],placeholder,p.dtype,false,false);
    encode(state->maps[2],placeholder,p.dtype,true,true);
    encode(state->maps[3],placeholder,p.dtype,false,true);
  }catch(...){cudaFree(placeholder);throw;}
  check(cudaFree(placeholder),"SW tensor template placeholder free");
  return state;''')
s=s.replace('  CUtensorMap a{},b{},s{},t{};encode(a,A,state.dtype,true,false);encode(b,B,state.dtype,false,false);\n  encode(s,workspace,state.dtype,true,true);encode(t,static_cast<char*>(workspace)+WorkspaceBytes/2,state.dtype,false,true);','''  CUtensorMap a=state.maps[0],b=state.maps[1],s=state.maps[2],t=state.maps[3];
  auto bind=[](CUtensorMap& map,const void* pointer){
    if(reinterpret_cast<uintptr_t>(pointer)%16)throw std::invalid_argument("SW TMA input must be16-byte aligned");
    auto result=cuTensorMapReplaceAddress(&map,const_cast<void*>(pointer));
    if(result!=CUDA_SUCCESS)throw std::runtime_error("SW tensor map replace status "+std::to_string(int(result)));
  };
  bind(a,A);bind(b,B);bind(s,workspace);bind(t,static_cast<char*>(workspace)+WorkspaceBytes/2);''')
assert 'encode(a,A' not in s
Path('solution/sw_pipeline.cu').write_text(s)
s=subprocess.check_output(['git','show','13340d1:solution/plan.cpp'],text=True).replace('current_pointer_maps=4','current_pointer_maps=4;host_geometry_templates=immutable;bind=cuTensorMapReplaceAddress')
Path('solution/plan.cpp').write_text(s)
