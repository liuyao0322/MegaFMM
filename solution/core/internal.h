#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
void run_sw_pipeline(const Problem& p,const void* A,const void* B,const float* C0,float* D,cudaStream_t stream);
size_t sw_hopper_workspace(const Problem& p);
void setup_sw_hopper(const Problem& p);
void run_sw_hopper(const Problem& p,const void* A,const void* B,float* D,
                   void* workspace,cudaStream_t stream);
const char* sw_hopper_config(const Problem& p);
// Metadata shared by the small general-purpose SW candidate pool.
struct SWConfig {int id, n, bk, stages, pack_cap, raster; bool padded; int groups=1;};
struct SWResources {size_t shared;int registers,resident;};
void run_sw_fp32(const Problem& p,const float* A,const float* B,const float* C,
                 float* D,cudaStream_t stream,int bk=16);
SWResources setup_sw_fp32(int bk);
std::vector<SWConfig> sw_configs(const Problem& p);
size_t sw_config_workspace(const Problem& p,const SWConfig& c);
SWResources setup_sw_config(const Problem& p,const SWConfig& c);
void run_sw_config(const Problem& p,const SWConfig& c,const void* A,const void* B,
                   float* D,void* workspace,cudaStream_t stream);
std::string sw_config_name(const SWConfig& c);
struct SWPrecision {bool pass;double relative_error,dense_error,limit;};
// A prepared original-input FP64 reference is reused across candidates, never across tune calls.
struct SWReference;
std::shared_ptr<SWReference> make_sw_reference(const Problem& p,const void* A,const void* B,
                                               const float* dense,cudaStream_t stream);
SWPrecision check_sw_reference(const SWReference& ref,const float* D,cudaStream_t stream);
bool dense_agreement(const Problem& p,const float* D,const float* reference,cudaStream_t stream);
// Original rounded-input FP64 precision guard (also exercised by regression tests).
bool original_fp64_agreement(const Problem& p,const void* A,const void* B,
                             const float* D,const float* dense,cudaStream_t stream);
}
