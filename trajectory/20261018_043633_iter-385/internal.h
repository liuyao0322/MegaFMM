#pragma once
#include "megafmm/gemm.h"
namespace megafmm::detail {
void scale_output(const Problem& p, const float* C0, float* D, cudaStream_t stream);
void run_sw_pipeline(const Problem& p,const void* A,const void* B,const float* C0,float* D,cudaStream_t stream);
// Restored verbatim from iter313 (79dad0b); explicit 1024 SW specialization.
struct SWState;
std::shared_ptr<SWState> make_sw_pipeline(const Problem& p);
void run_sw_pipeline(const SWState& state,const void* A,const void* B,float* D,
                     void* workspace,cudaStream_t stream);
size_t sw_hopper_workspace(const Problem& p);
void setup_sw_hopper(const Problem& p);
void run_sw_hopper(const Problem& p,const void* A,const void* B,float* D,
                   void* workspace,cudaStream_t stream);
const char* sw_hopper_config(const Problem& p);
// Native configurations: architecture tuple + runtime packing/raster schedule.
struct SWConfig {int id, n, bk, stages, pack_cap, raster; bool padded;};
struct SWResources {size_t shared;int registers,resident;};
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
constexpr size_t SW1024WorkspaceBytes=size_t(4)<<20;
constexpr int HopperCandidateCount=112;
bool wmma_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                 cudaStream_t stream,int operation);
bool mma_even_dense(int bk,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    cudaStream_t stream,int operation);
bool ampere_dense(int id,const Problem& p,const void* A,const void* B,const float* C,float* D,
                  cudaStream_t stream,int operation);
size_t hopper_workspace(const Problem& p,int id,int sm_count=0);
size_t hopper_streamk_workspace(const Problem& p,int id,int sm_count);
bool padded_hopper(int base,const Problem& p,const void* A,const void* B,const float* C,float* D,
                    void* workspace,int sm_count,cudaStream_t stream,int operation);
bool dense_agreement(const Problem& p,const float* D,const float* reference,cudaStream_t stream);
// Explicit-tuning fallback for beta-zero K-reduction candidates. Uses original
// rounded inputs and the unchanged FP64-relative gate, not Dense agreement.
bool original_fp64_agreement(const Problem& p,const void* A,const void* B,
                             const float* D,const float* dense,cudaStream_t stream);
const char* hopper_candidate_name(int id);
bool hopper_dense(int id,const Problem& p,const void* A,const void* B,const float* C0,float* D,
                  void* workspace,int sm_count,cudaStream_t stream,int operation);
}
