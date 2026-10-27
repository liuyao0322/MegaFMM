#pragma once
#include "internal.h"
#include <array>
#include <cuda.h>
#include <tuple>
#include <type_traits>
#include <utility>

namespace megafmm::detail::sw_fp32_batch {
// Materialized: one product per CTA, global products, separate reconstruction.
// CooperativeFused: product warp groups + one issuer warp, shared reconstruction;
// cooperative packing within actual occupancy capacity. PersistentTasks optionally
// bounds the resident grid and loops over output tasks; otherwise pack+fused.
enum class ExecutionStrategy {Materialized,CooperativeFused,DirectBMaterialized};
// HybridRawB retains the existing three raw B products + four packed transforms,
// including its fully-packed binding fallback. RawQuadrants is a different
// policy: a producer warp reads all four original quadrants and forms seven terms.
enum class OperandSource {PackedTransforms,PackedQuadrants,RawQuadrants,HybridRawB};
enum class PackExecution {Automatic,SeparateKernel,CooperativeResident};
enum class TailPolicy {Padded,GuardedFinalTile};
enum class ProducerPolicy {TmaIssuer,WarpTransform,WarpTransformTmaB};
// Float4-preserving permutation for the shared seven-product handoff.
enum class ProductLayout {Linear,RowXor4};
template<OperandSource A=OperandSource::PackedTransforms,
         OperandSource B=OperandSource::PackedTransforms,
         PackExecution Pack=PackExecution::Automatic,
         TailPolicy Tail=TailPolicy::Padded,
         ProducerPolicy Producer=ProducerPolicy::TmaIssuer>
struct DataflowPolicy {
  static_assert(A==OperandSource::PackedTransforms || A==OperandSource::PackedQuadrants || A==OperandSource::RawQuadrants,"invalid A source");
  static_assert(B==OperandSource::PackedTransforms || B==OperandSource::PackedQuadrants || B==OperandSource::RawQuadrants || B==OperandSource::HybridRawB,"invalid B source");
  static_assert(Pack==PackExecution::Automatic || Pack==PackExecution::SeparateKernel || Pack==PackExecution::CooperativeResident,"invalid packing execution");
  static_assert(Tail==TailPolicy::Padded || Tail==TailPolicy::GuardedFinalTile,"invalid tail policy");
  static_assert(Producer==ProducerPolicy::TmaIssuer || Producer==ProducerPolicy::WarpTransform || Producer==ProducerPolicy::WarpTransformTmaB,"invalid producer policy");
  static_assert(Producer!=ProducerPolicy::WarpTransformTmaB || (B==OperandSource::RawQuadrants && A!=OperandSource::RawQuadrants),"TMA B producer requires raw B quadrants and packed A descriptor workspace");
  static constexpr auto a_source=A,b_source=B;
  static constexpr auto pack_execution=Pack;
  static constexpr auto tail_policy=Tail;
  static constexpr auto producer_policy=Producer;
};
inline constexpr const char* strategy_name(ExecutionStrategy s) {
  return s==ExecutionStrategy::Materialized?"materialized":s==ExecutionStrategy::CooperativeFused?"cooperative_fused":"direct_b_materialized";
}
template<int BM,int BN,int BK,int TM,int TN,int WarpsM,int WarpsN,int ComputeK,int GroupM,
         ExecutionStrategy Strategy=ExecutionStrategy::Materialized,bool PersistentTasks=false,int ProgrammaticMask=0,bool ZigzagFma=false,bool RetainProducts=false,bool FusedRawB=false,int FusedSlots=2,
         class Flow=DataflowPolicy<OperandSource::PackedTransforms,
           (FusedRawB || Strategy==ExecutionStrategy::DirectBMaterialized)?OperandSource::HybridRawB:OperandSource::PackedTransforms>,
         ProductLayout Products=ProductLayout::Linear>
struct KernelConfig {
  using dataflow=Flow;
  static constexpr int output_tail_rows=0;
  static constexpr bool vector_packing=false,vector_pack_b=false,compact_pack=false;
  static constexpr bool interleaved_operands=false,lane_reconstruct=false;
  static constexpr auto product_layout=Products;
  static_assert(Products==ProductLayout::Linear || Products==ProductLayout::RowXor4,"invalid product layout");
  static_assert(Products==ProductLayout::Linear || (Strategy==ExecutionStrategy::CooperativeFused && BN%8==0),"row XOR requires a fused eight-element column span");
  static constexpr auto a_source=Flow::a_source,b_source=Flow::b_source;
  static constexpr auto pack_execution=Flow::pack_execution;
  static constexpr auto tail_policy=Flow::tail_policy;
  static constexpr auto producer_policy=Flow::producer_policy;
  static constexpr bool raw_b_tma=producer_policy==ProducerPolicy::WarpTransformTmaB;
  static constexpr bool warp_transform=producer_policy==ProducerPolicy::WarpTransform || raw_b_tma;
  static constexpr bool policy_pack=a_source!=OperandSource::PackedTransforms ||
    b_source==OperandSource::PackedQuadrants || b_source==OperandSource::RawQuadrants;
  static constexpr int a_planes=a_source==OperandSource::RawQuadrants?0:a_source==OperandSource::PackedQuadrants?4:7;
  static constexpr int b_planes=b_source==OperandSource::RawQuadrants?0:b_source==OperandSource::PackedQuadrants?4:7;
  static_assert(TM>0 && TN>0 && TN%4==0 &&
    (TM%4==0 || (Strategy==ExecutionStrategy::CooperativeFused && TM==2)),
    "thread tiles require float4 columns; fused rows additionally support float2");
  static_assert(WarpsM>0 && WarpsN>0 && 32*WarpsM*WarpsN<=1024,"invalid warp grid");
  static constexpr auto strategy=Strategy;
  static constexpr bool persistent_tasks=PersistentTasks;
  static_assert(ProgrammaticMask>=0 && ProgrammaticMask<=3,"invalid PDL edge mask");
  static constexpr int pdl_mask=ProgrammaticMask;
  static constexpr bool zigzag_fma=ZigzagFma;
  static constexpr bool retain_products=RetainProducts;
  static constexpr bool fused_raw_b=Strategy==ExecutionStrategy::CooperativeFused && b_source==OperandSource::HybridRawB;
  static_assert(!FusedRawB || fused_raw_b,"legacy rawB flag conflicts with dataflow policy");
  static constexpr int fused_slots=FusedSlots;
  static_assert(FusedSlots>=2 && FusedSlots<=4,"invalid fused stage count");
  static_assert(FusedSlots==2 || fused_raw_b || warp_transform,"additional stages require rawB or warp producer");
  static constexpr bool pdl_pack=(ProgrammaticMask&1)!=0,pdl_combine=(ProgrammaticMask&2)!=0;
  static_assert(ProgrammaticMask==0 || Strategy==ExecutionStrategy::DirectBMaterialized ||
    (Strategy==ExecutionStrategy::CooperativeFused && ProgrammaticMask==1 && Flow::pack_execution==PackExecution::SeparateKernel),
    "PDL requires directB materialized leaves or a separate-pack fused input edge");
  static constexpr bool fused=Strategy==ExecutionStrategy::CooperativeFused;
  static constexpr bool direct_b=Strategy==ExecutionStrategy::DirectBMaterialized;
  static_assert(!policy_pack || (fused && warp_transform),"quadrant sources require a fused warp producer");
  static_assert(!warp_transform || fused,"warp producer requires fusion");
  static_assert(fused || tail_policy==TailPolicy::Padded,"explicit tail policy requires fusion; materialized leaves already guard K");
  static_assert(fused || pack_execution!=PackExecution::CooperativeResident,"cooperative packing requires fusion");
  static_assert(!direct_b || b_source==OperandSource::HybridRawB,"directB strategy requires hybrid B source");
  static_assert(fused || direct_b || b_source==OperandSource::PackedTransforms,"materialized strategy requires packed transforms");
  static_assert(fused ? (BN%(TN*WarpsN)==0 && 32%(BN/(TN*WarpsN))==0 && BM==TM*WarpsM*(32/(BN/(TN*WarpsN))))
                     : (BM==8*TM*WarpsM && BN==4*TN*WarpsN),"tile must match strategy lane grid");
  static_assert(BM>0 && BN>0 && BM%4==0 && BN%4==0 && BM<=256 && BN<=256,"invalid TMA tile");
  static_assert(fused || (BM%32==0 && BN%32==0),"materialized output requires packed tiles");
  static_assert(!fused || GroupM==1,"fused strategy currently uses row-major CTA order");
  static_assert(BK==16 || BK==32 || ((fused_raw_b || warp_transform) && BK==8),"invalid shared packed K tile");
  static_assert(ComputeK>0 && BK%ComputeK==0,"ComputeK must divide BK");
  static_assert(GroupM>0,"GroupM must be positive; 1 selects row-major order");
  static constexpr int bm=BM,bn=BN,bk=BK,tm=TM,tn=TN;
  static constexpr int warps_m=WarpsM,warps_n=WarpsN;
  static constexpr int product_warps=WarpsM*WarpsN;
  static constexpr int compute_warps=fused?7*product_warps:product_warps;
  static constexpr int threads=fused?32*(compute_warps+1):32*product_warps;
  static constexpr int min_blocks=fused && TM*TN<=16?2:1;
  static_assert(threads<=1024,"product warps and issuer exceed CTA size");
  static constexpr int compute_k=ComputeK,group_m=GroupM;
  static constexpr int max_task_waves=0;
};
// Restrict tuning enumeration to a tile workload without changing kernel legality.
// A wave means one output-task CTA per SM; materialized products count separately.
template<class Kernel,int MaxTaskWaves> struct TuningConfig:Kernel {
  static_assert(MaxTaskWaves>0,"tuning task-wave limit must be positive");
  static constexpr int max_task_waves=MaxTaskWaves;
};
// Guarded coalesced scalar input loads and vector packed output stores.
// A uses its required shared transpose; B optionally uses shared regrouping.
template<class Kernel,bool VectorB,bool Compact> struct PackingConfig:Kernel {
  static_assert((Kernel::a_planes==4 || Kernel::a_planes==7) && (Kernel::b_planes==4 || Kernel::b_planes==7),"vector packing requires stored operand planes");
  static constexpr bool vector_packing=true,vector_pack_b=VectorB,compact_pack=Compact;
};
// A partial final output tile can omit rows no valid output can consume.
// The decision is uniform across each CTA and leaves every barrier arrival intact.
template<class Kernel,int Rows> struct OutputTailConfig:Kernel {
  static_assert(Kernel::fused,"output row-group guard requires fusion");
  static_assert(Rows>0 && Rows<Kernel::tm,"row-group guard must trim the thread tile");
  static constexpr int output_tail_rows=Rows;
};
enum class RegisterGather { StreamedSeven, SelectedSeven, SelectedFour };
// Eight physical slots hold the original seven rounded transforms. Shuffle
// ownership uses every compute lane; only six cross-warp microtiles spill.
template<class Kernel,bool Shuffle> struct ReconstructionConfig:Kernel {
  static_assert(Kernel::fused && Kernel::warp_transform &&
    Kernel::a_source==OperandSource::PackedTransforms && Kernel::b_source==OperandSource::PackedTransforms,
    "interleaved reconstruction requires packed transforms and a warp producer");
  static_assert(Kernel::pack_execution==PackExecution::SeparateKernel && !Kernel::persistent_tasks,
    "interleaved reconstruction requires separate ordinary packing");
  static_assert(Kernel::bm==64 && Kernel::bn==32 && Kernel::tm==8 && Kernel::tn==8 && Kernel::product_warps==1,
    "interleaved reconstruction requires the 32-position eight-by-eight microtile grid");
  static_assert(Kernel::output_tail_rows==0 && Kernel::product_layout==ProductLayout::Linear,
    "interleaved reconstruction owns its output layout");
  static_assert(Kernel::tail_policy==TailPolicy::Padded, "interleaved reconstruction requires padded K tiles");
  static constexpr bool vector_packing=true,vector_pack_b=true,compact_pack=true;
  static constexpr bool coalesced_pack=false;
  static constexpr RegisterGather register_gather=RegisterGather::StreamedSeven;
  static constexpr bool interleaved_operands=true,lane_reconstruct=Shuffle;
  static constexpr int a_planes=8,b_planes=8,min_blocks=2;
};
// Register source selection and output scheduling are independent of input
// shape and packing. Only a warp-owned reconstruction may select this policy.
template<class Kernel,RegisterGather Gather> struct GatherConfig:Kernel {
  static_assert(Kernel::interleaved_operands && Kernel::lane_reconstruct,
    "register gather requires interleaved shuffle reconstruction");
  static constexpr RegisterGather register_gather=Gather;
};
template<class Kernel,bool Coalesced=true> struct PackOwnershipConfig:Kernel {
  static_assert(Kernel::interleaved_operands, "slot packing requires interleaved operands");
  static constexpr bool coalesced_pack=Coalesced;
};
// This is the only registration list. Every entry supports all eligible shapes;
// SWTuned measures eligible entries; optional task-wave limits bound tuning work.
using KernelConfigurations=std::tuple<
#ifdef MEGAFMM_FP32_CONFIG_HEADER
#include MEGAFMM_FP32_CONFIG_HEADER
#else
// Retain accepted policies; bound additional tuning by output task waves.
KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized,false,3,true,true>,
KernelConfig<16,32,32,4,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>,
KernelConfig<64,32,32,8,8,1,1,32,1,ExecutionStrategy::CooperativeFused>,
KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized>,
KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true>,
KernelConfig<128,32,8,8,8,2,1,8,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,4>,
KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,2,
  DataflowPolicy<OperandSource::PackedQuadrants,OperandSource::HybridRawB,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,
KernelConfig<64,32,32,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,2,
  DataflowPolicy<OperandSource::PackedQuadrants,OperandSource::HybridRawB,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,
KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,2,
 DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB>,ProductLayout::RowXor4>,
KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,2,
  DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::Automatic,TailPolicy::GuardedFinalTile>>,
KernelConfig<32,32,16,4,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true,2,
  DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::Automatic,TailPolicy::GuardedFinalTile>>,
TuningConfig<KernelConfig<32,8,32,2,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true,0,true>,1>,
TuningConfig<KernelConfig<128,32,32,8,8,2,1,16,4,ExecutionStrategy::DirectBMaterialized,false,2,false>,2>,
TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>,ProductLayout::Linear>,8>,
TuningConfig<KernelConfig<64,32,32,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>,ProductLayout::Linear>,8>,
TuningConfig<KernelConfig<64,64,32,8,8,1,2,16,4,ExecutionStrategy::DirectBMaterialized,false,3,true,true>,32>,
TuningConfig<KernelConfig<64,32,32,8,8,1,1,16,4,ExecutionStrategy::DirectBMaterialized,false,3,true,true>,32>,
TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,true,2,DataflowPolicy<OperandSource::PackedQuadrants,OperandSource::HybridRawB,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,
TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>,8>,
PackingConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedQuadrants,PackExecution::SeparateKernel,TailPolicy::GuardedFinalTile,ProducerPolicy::WarpTransform>,ProductLayout::RowXor4>,8>,true,true>,
OutputTailConfig<PackingConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedQuadrants,PackExecution::SeparateKernel,TailPolicy::GuardedFinalTile,ProducerPolicy::WarpTransform>,ProductLayout::RowXor4>,8>,true,true>,1>
#endif
>;

struct Config {
  int id,bm,bn,bk,tm,tn,warps_m,warps_n,threads,compute_k,group_m;
  ExecutionStrategy strategy;bool persistent_tasks;int pdl_mask;
  bool zigzag_fma,retain_products,fused_raw_b;int fused_slots;
  OperandSource a_source,b_source;PackExecution pack_execution;
  ProductLayout product_layout;
  TailPolicy tail_policy;ProducerPolicy producer_policy;int a_planes,b_planes,max_task_waves;
  bool vector_packing,vector_pack_b,compact_pack;int output_tail_rows;
};
template<size_t I> constexpr Config configuration() {
  using C=std::tuple_element_t<I,KernelConfigurations>;
  return {int(I),C::bm,C::bn,C::bk,C::tm,C::tn,C::warps_m,C::warps_n,C::threads,C::compute_k,C::group_m,C::strategy,C::persistent_tasks,C::pdl_mask,C::zigzag_fma,C::retain_products,C::fused_raw_b,C::fused_slots,C::a_source,C::b_source,C::pack_execution,C::product_layout,C::tail_policy,C::producer_policy,C::a_planes,C::b_planes,C::max_task_waves,C::vector_packing,C::vector_pack_b,C::compact_pack,C::output_tail_rows};
}
template<size_t... I> constexpr auto configuration_list(std::index_sequence<I...>) {
  return std::array<Config,sizeof...(I)>{{configuration<I>()...}};
}
inline constexpr auto configurations=configuration_list(
  std::make_index_sequence<std::tuple_size_v<KernelConfigurations>>{});
inline constexpr bool tuning_eligible(const Config& c,int64_t m,int64_t n,int sms) {
  if(c.max_task_waves==0)return true;
  if(m<=0 || n<=0 || sms<=0)return false;
  const int64_t mt=(((m+1)/2)+c.bm-1)/c.bm;
  const int64_t nt=(((n+1)/2)+c.bn-1)/c.bn;
  const int products=c.strategy==ExecutionStrategy::CooperativeFused?1:7;
  return mt*nt*products<=int64_t(sms)*c.max_task_waves;
}
struct Shape {int mp,np,kp;size_t a,b,products,bytes;};
struct alignas(64) Maps {CUtensorMap a{},b{},raw_b{},rest_b{};const void* workspace=nullptr;int id=-1,mp=0,np=0,kp=0;int cooperative_max_blocks=0;bool raw_b_available=false;int raw_n=0,raw_k=0,raw_ldb=0;};
Maps prepare(const Problem& p,int id,void* workspace);
void run_prepared(const Problem& p,int id,const float* A,const float* B,const float* C,
                  float* D,void* workspace,cudaStream_t stream,const Maps& maps);
bool problem(const Problem& p);
Shape shape(const Problem& p);
Shape shape(const Problem& p,int id);
size_t workspace_size(const Problem& p,int id);
// Binding-dependent route; no allocation or device synchronization.
bool uses_cooperative(const Problem& p,int id,const float* A,const float* B,const Maps& maps);
bool uses_direct_b(const Problem& p,int id,const float* A,const float* B,const Maps& maps);
SWResources setup(int id);
std::string name(int id);
void run(const Problem& p,int id,const float* A,const float* B,const float* C,
         float* D,void* workspace,cudaStream_t stream);
}
