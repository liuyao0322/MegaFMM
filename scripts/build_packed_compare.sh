#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/build.sh
bash scripts/on_gpu.sh bash -s -- "${MEGAFMM_BUILD_DIR:-build}" <<'BUILD'
set -euo pipefail
build_dir=$1
for entry in baseline:9d051d0a previous:33ec1bc8 archived:3a38f6eb; do
  role=${entry%%:*}
  revision=${entry#*:}
  reference_dir="$build_dir/packed_$role"
  mkdir -p "$reference_dir"
  source_prefix=solution/sw/fp32
  config_id=0
  if [[ "$role" == archived ]]; then
    source_prefix=archive/sw_specializations/solution/sw/fp32
    config_id=117
  fi
  git show "$revision:$source_prefix/sw_fp32_batch.cu" > "$reference_dir/sw_fp32_batch.cu"
  git show "$revision:$source_prefix/sw_fp32_batch.h" > "$reference_dir/sw_fp32_batch.h"
  if [[ "$role" == archived ]]; then
    # Freeze all device bodies verbatim. Restrict only the host dispatch to 117
    # so the benchmark does not instantiate the other 153 archived candidates.
    python3 - "$reference_dir/sw_fp32_batch.cu" <<'TRIM'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
a=s.index('int cooperative_resident(int id) {');b=s.index('\nSWResources select(',a)
s=s[:a]+"""int cooperative_resident(int id) {
  int resident=0;
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
    cooperative_fused_leaves<16,32,4,4,32,true,2,false,true>,256,sizeof(FusedShared<16,32,32>)));
  return resident;
}
"""+s[b:]
a=s.index('SWResources select(');b=s.index('\n}\n}\nbool problem(',a)
brace=s.index('{',a)
s=s[:brace]+"""{
  if(id!=117)throw std::invalid_argument("frozen control only supports 117");
  return cooperative_dispatch<16,32,4,4,32,true,2,false,true>(p,A,B,C,D,workspace,stream,prepared);
"""+s[b:]
p.write_text(s)
TRIM
  fi
  cat > "$reference_dir/bridge.cu" <<BRIDGE
#define sw_fp32_batch ${role}_fp32_batch
#include "sw_fp32_batch.cu"
namespace frozen=megafmm::detail::${role}_fp32_batch;
extern "C" size_t megafmm_${role}_workspace(const megafmm::Problem* p){return frozen::shape(*p).bytes;}
extern "C" void* megafmm_${role}_prepare(const megafmm::Problem* p,void* w){frozen::setup($config_id);return new frozen::Maps(frozen::prepare(*p,$config_id,w));}
extern "C" void megafmm_${role}_destroy(void* maps){delete static_cast<frozen::Maps*>(maps);}
extern "C" void megafmm_${role}_run(const megafmm::Problem* p,const float* a,const float* b,float* d,void* w,cudaStream_t s,void* maps){frozen::run_prepared(*p,$config_id,a,b,nullptr,d,w,s,*static_cast<frozen::Maps*>(maps));}
BRIDGE
  nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
    -Iinclude -Isolution/core -Ithird_party/cutlass/include \
    -c "$reference_dir/bridge.cu" -o "$reference_dir/bridge.o"
done
config_flags=()
if [[ -n "${MEGAFMM_FP32_CONFIG_HEADER:-}" ]]; then
  config_flags+=("-DMEGAFMM_FP32_CONFIG_HEADER=\"$(realpath "$MEGAFMM_FP32_CONFIG_HEADER")\"")
fi
nvcc -O3 -std=c++17 -arch=sm_90a "${config_flags[@]}" \
  -Iinclude -Isolution/core -Isolution/sw/fp32 \
  bench/packed_fp32_compare.cu "$build_dir/packed_baseline/bridge.o" \
  "$build_dir/packed_previous/bridge.o" "$build_dir/packed_archived/bridge.o" "$build_dir/libmegafmm.a" \
  -lcublas -lcublasLt -lcuda -o "$build_dir/packed_fp32_compare"
BUILD
