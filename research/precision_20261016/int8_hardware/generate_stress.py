"""Reuse the accepted comparator's extended cases, guards and dual-stream stress."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r.parent/'validate_direct.cu').read_text()
s=s[:s.index('int main(')]
s=s.replace('../../tests/test_sw1024.cu','../../../tests/test_sw1024.cu')
a=s.index('#ifndef DIRECT_HEADER');b=s.index('#include <cuda_profiler_api.h>')
s=s[:a]+'#include "corrected.cuh"\n'+s[b:]
s=s.replace('Graph graph[2];','Graph graph[2];\n Buffer workspace[2]={Buffer(int8_research::WorkBytes),Buffer(int8_research::WorkBytes)};')
old='precision_research::run(dtype,a[s].p,b[s].p,d[s].as<float>()+128,ss[s].s);'
new='int8_research::run(dtype,a[s].p,b[s].p,d[s].as<float>()+128,workspace[s].p,ss[s].s,true);'
assert s.count(old)==2
s=s.replace(old,new)
s=s.replace('int validate_extra(megafmm::DType dtype){','int validate_extra(megafmm::DType dtype,bool quick){')
s=s.replace('trial<3','trial<(quick?1:3)').replace('repeat<8','repeat<(quick?2:8)')
s+='''int main(int argc,char** argv){try{
 bool quick=argc==2&&std::string(argv[1])=="--sanitizer";
 if(argc!=1&&!quick)throw std::invalid_argument("usage: stress [--sanitizer]");
 int8_research::init();
 int failures=validate_extra<__nv_bfloat16>(megafmm::DType::BF16,quick)+validate_extra<__half>(megafmm::DType::FP16,quick);
 return failures?2:0;
}catch(const std::exception&e){fprintf(stderr,"%s\\n",e.what());return 1;}}
'''
(r/'stress.cu').write_text('// Generated from prior extended validation; original reference unchanged.\n'+s)
