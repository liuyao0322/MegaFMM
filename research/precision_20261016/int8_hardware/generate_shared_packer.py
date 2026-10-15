"""Generate a shared-staged two-plane producer from a measured register candidate.

Preserve all numerical operations/layouts and the base consumer. The experiment
uses bounded unrolling and dynamic shared storage to reduce live register arrays.
"""
import argparse
import json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);args=ap.parse_args()
h=Path(__file__).resolve().parent;base=h/args.base
j=json.loads((base/f'{args.base}_balanced.json').read_text());assert len(j['checks'])==78 and all(x['pass']=='1' for x in j['checks'])
s=(base/'pack.cuh').read_text()
s=s.replace('struct PairPackShared {float maxima[8];int exponent;int8_t transpose[2][128*36];};','template<bool SideA> struct PairPackShared {float values[2][SideA?8192:4096];float maxima[8];int exponent;int8_t transpose[SideA?1:2][SideA?1:128*36];};')
s=s.replace('float (&v)[(SideA?8192:4096)/256],T* high','int which,T* high').replace('PairPackShared& s','PairPackShared<SideA>& s')
s=s.replace('T h=T(v[j]);','T h=T(s.values[which][i]);')
a=s.index(' // Recompute cast/residual');b=s.index(' auto quantize=',a);s=s[:a]+s[b:]
s=s.replace('int i=tid+256*j;float h=float(T(v[j]));','int i=tid+256*j;float v=s.values[which][i],h=float(T(v));').replace('__fsub_rn(v[j],h)','__fsub_rn(v,h)')
s=s.replace(' float v0[PT],v1[PT];\n','').replace('v0[j]','s.values[0][i]').replace('v1[j]','s.values[1][i]')
s=s.replace('(v0,high,','(0,high,').replace('(v1,high,','(1,high,')
s=s.replace(' }else load(std::false_type{},0,0);\n if constexpr(Group==0)', ' }else load(std::false_type{},0,0);\n __syncthreads();\n if constexpr(Group==0)')
a=s.index('// Keep only two FP32 transformed planes');b=s.index('template<class T,bool Balance> void preprocess',a)
region=s[a:b].replace(' #pragma unroll\n',' #pragma unroll 1\n')
region=region.replace(' __shared__ PairPackShared scratch;',' extern __shared__ __align__(128) unsigned char scratch_buffer[];\n auto& scratch=*reinterpret_cast<PairPackShared<SideA>*>(scratch_buffer);')
s=s[:a]+region+s[b:]
s=s.replace('<<<3*32,256,0,stream>>>','<<<3*32,256,sizeof(PairPackShared<true>),stream>>>').replace('<<<3*64,256,0,stream>>>','<<<3*64,256,sizeof(PairPackShared<false>),stream>>>')
assert 'v[j]' not in s[a:] and 'v0' not in s[a:]
(h/'pack.cuh').write_text(s)
s=(base/'compare.cu').read_text().replace('<<<3*32,256,0,stream.s>>>','<<<3*32,256,sizeof(int8_research::PairPackShared<true>),stream.s>>>').replace('<<<3*64,256,0,stream.s>>>','<<<3*64,256,sizeof(int8_research::PairPackShared<false>),stream.s>>>');(h/'compare.cu').write_text(s)
s=(base/'corrected.cuh').read_text();needle='void init(){';assert s.count(needle)==1
helper='''template<class T,bool Balance> void init_shared_packer(){
 check(cudaFuncSetAttribute(pack<T,true,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<true>)),"pair A dynamic shared");
 check(cudaFuncSetAttribute(pack<T,false,Balance>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PairPackShared<false>)),"pair B dynamic shared");
}
void init(){
 init_shared_packer<cute::half_t,true>();init_shared_packer<cute::half_t,false>();
 init_shared_packer<cute::bfloat16_t,true>();init_shared_packer<cute::bfloat16_t,false>();'''
s=s.replace(needle,helper);(h/'corrected.cuh').write_text(s)
print('generated shared packer from',args.base)
