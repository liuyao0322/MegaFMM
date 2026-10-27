#!/usr/bin/env python3
"""Compile the dataflow contract, including combinations that must be rejected."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cuda-include", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [args.cxx, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"]
    for include in (root / "include", root / "solution/core",
                    root / "solution/sw/fp32", Path(args.cuda_include)):
        command.extend(["-I", str(include)])
    prefix = '#include "sw_fp32_batch.h"\nusing namespace megafmm::detail::sw_fp32_batch;\n'
    fused = "16,32,16,4,4,1,1,8,1,ExecutionStrategy::CooperativeFused,true,0,false,false,false,2,"
    materialized = "64,64,16,4,8,2,2,16,1,ExecutionStrategy::Materialized,false,0,false,false,false,2,"
    cases = {
        "legacy_defaults": (True, """
using C = KernelConfig<16,32,32,4,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>;
static_assert(C::a_planes==7 && C::b_planes==7);
static_assert(C::pack_execution==PackExecution::Automatic && !C::warp_transform);
static_assert(C::tail_policy==TailPolicy::Padded);
static_assert(C::product_layout==ProductLayout::Linear);
"""),
        "legacy_raw_b": (True, """
using C = KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true>;
static_assert(C::b_source==OperandSource::HybridRawB && C::fused_raw_b);
"""),
        "quadrants_with_issuer": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::PackedQuadrants>>;"),
        "raw_with_issuer": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::RawQuadrants>>;"),
        "materialized_warp": (False, f"using C=KernelConfig<{materialized}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>;"),
        "materialized_cooperative": (False, f"using C=KernelConfig<{materialized}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::CooperativeResident>>;"),
        "materialized_explicit_tail": (False, f"using C=KernelConfig<{materialized}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::GuardedFinalTile>>;"),
        "hybrid_a": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::HybridRawB>>;"),
        "invalid_pack_enum": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,static_cast<PackExecution>(99)>>;"),
        "legacy_raw_conflict": (False, "using C=KernelConfig<16,32,16,4,4,1,1,8,1,ExecutionStrategy::CooperativeFused,false,0,false,false,true,2,DataflowPolicy<>>;"),
    }
    cases.update({
        "fused_pack_pdl": (True, "using C=KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,true,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::SeparateKernel>>;static_assert(C::pdl_pack && !C::pdl_combine);"),
        "fused_automatic_pdl": (False, "using C=KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,true>;"),
        "fused_combine_pdl": (False, "using C=KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,2,true,false,true,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::SeparateKernel>>;"),
        "fused_both_edges_pdl": (False, "using C=KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,3,true,false,true,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::SeparateKernel>>;"),
    })
    cases.update({
        "fused_float2_rows": (True, "using C=KernelConfig<16,16,16,2,4,1,1,16,1,ExecutionStrategy::CooperativeFused,true,0,true>;static_assert(C::tm==2 && C::tn==4);"),
        "float2_rows_materialized": (False, "using C=KernelConfig<16,32,32,2,8,1,1,32,1,ExecutionStrategy::DirectBMaterialized>;"),
        "unsupported_scalar_rows": (False, "using C=KernelConfig<8,16,32,1,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>;"),
        "unsupported_three_rows": (False, "using C=KernelConfig<12,32,32,3,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>;"),
        "unsupported_narrow_columns": (False, "using C=KernelConfig<16,8,32,2,2,1,1,32,1,ExecutionStrategy::CooperativeFused,true>;"),
    })
    cases.update({
        "fused_product_xor": (True, f"using C=KernelConfig<{fused}DataflowPolicy<>,ProductLayout::RowXor4>;static_assert(C::product_layout==ProductLayout::RowXor4);"),
        "materialized_product_xor": (False, f"using C=KernelConfig<{materialized}DataflowPolicy<>,ProductLayout::RowXor4>;"),
        "invalid_product_enum": (False, f"using C=KernelConfig<{fused}DataflowPolicy<>,static_cast<ProductLayout>(99)>;"),
        "product_xor_column_span": (False, "using C=KernelConfig<128,12,16,4,4,1,3,8,1,ExecutionStrategy::CooperativeFused,false,0,false,false,false,2,DataflowPolicy<>,ProductLayout::RowXor4>;"),
    })
    cases.update({
        "tma_b_raw_quadrants": (True, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>;static_assert(C::raw_b_tma && C::warp_transform && C::b_planes==0);"),
        "tma_b_packed_a4": (True, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::PackedQuadrants,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>;static_assert(C::a_planes==4 && C::b_planes==0);"),
        "tma_b_hybrid_source": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::HybridRawB,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>;"),
        "tma_b_no_descriptor_workspace": (False, f"using C=KernelConfig<{fused}DataflowPolicy<OperandSource::RawQuadrants,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>;"),
        "tma_b_materialized": (False, f"using C=KernelConfig<{materialized}DataflowPolicy<OperandSource::PackedTransforms,OperandSource::RawQuadrants,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransformTmaB>>;"),
        "legacy_no_tma_b": (True, f"using C=KernelConfig<{fused}DataflowPolicy<>>;static_assert(!C::raw_b_tma);"),
    })
    cases.update({
        "bounded_tuning_config": (True, "using C=TuningConfig<KernelConfig<32,8,32,2,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true,0,true>,1>;static_assert(C::max_task_waves==1 && C::tm==2);"),
        "zero_tuning_bound": (False, "using C=TuningConfig<KernelConfig<16,32,32,4,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>,0>;"),
        "negative_tuning_bound": (False, "using C=TuningConfig<KernelConfig<16,32,32,4,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>,-1>;"),
        "fused_tuning_wave_boundary": (True, """
using C=KernelConfig<32,8,32,2,4,1,1,32,1,ExecutionStrategy::CooperativeFused,true>;
constexpr auto cfg=[] { auto c=configuration<0>();c.strategy=ExecutionStrategy::CooperativeFused;c.bm=32;c.bn=8;c.max_task_waves=1;return c;}();
static_assert(tuning_eligible(cfg,256,256,132));
static_assert(!tuning_eligible(cfg,512,512,132));
static_assert(!tuning_eligible(cfg,256,256,0));
"""),
        "materialized_tuning_wave_boundary": (True, """
using C=KernelConfig<128,32,32,8,8,2,1,16,4,ExecutionStrategy::DirectBMaterialized>;
constexpr auto cfg=[] {auto c=configuration<0>();c.bm=128;c.bn=32;c.max_task_waves=2;return c;}();
static_assert(tuning_eligible(cfg,768,768,132));
static_assert(!tuning_eligible(cfg,769,768,132));
static_assert(!tuning_eligible(cfg,768,769,132));
static_assert(!tuning_eligible(cfg,513,8191,132));
static_assert(tuning_eligible(configuration<0>(),16384,16384,132));
"""),
    })
    cases.update({
        "coalesced_vector_a": (True, "using C=PackingConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true>,false,false>;static_assert(C::vector_packing && !C::vector_pack_b && !C::compact_pack);"),
        "coalesced_vector_ab": (True, "using C=PackingConfig<KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized>,true,true>;static_assert(C::vector_packing && C::vector_pack_b && C::compact_pack);"),
    })
    cases.update({
        "coalesced_packed_b4": (True, "using C=PackingConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedQuadrants,PackExecution::SeparateKernel,TailPolicy::GuardedFinalTile,ProducerPolicy::WarpTransform>>,true,true>;static_assert(C::b_planes==4 && C::warp_transform && C::pdl_pack && C::vector_packing);"),
    })
    cases.update({
        "output_single_row": (True, "using C=OutputTailConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true>,1>;static_assert(C::output_tail_rows==1 && C::tm==8);"),
        "output_four_rows": (True, "using C=OutputTailConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,0,true,false,true>,4>;static_assert(C::output_tail_rows==4);"),
        "output_zero_rows": (False, "using C=OutputTailConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused>,0>;"),
        "output_materialized_rows": (False, "using C=OutputTailConfig<KernelConfig<128,64,32,8,8,2,2,16,4,ExecutionStrategy::DirectBMaterialized>,1>;"),
    })
    cases.update({'physical_slot_packing_21': (True, 'using C=PackOwnershipConfig<ReconstructionConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,false>,true>;static_assert(C::coalesced_pack && C::threads==256 && C::compute_warps==7 && C::a_planes==8 && C::b_planes==8 && C::vector_packing && C::vector_pack_b && C::compact_pack);'), 'physical_slot_packing_22': (True, 'using C=PackOwnershipConfig<ReconstructionConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,8,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,false>,true>;static_assert(C::coalesced_pack && C::threads==256 && C::compute_warps==7 && C::a_planes==8 && C::b_planes==8 && C::vector_packing && C::vector_pack_b && C::compact_pack);'), 'physical_slot_packing_23': (True, 'using C=PackOwnershipConfig<GatherConfig<ReconstructionConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,true>,RegisterGather::SelectedFour>,true>;static_assert(C::coalesced_pack && C::threads==256 && C::compute_warps==7 && C::a_planes==8 && C::b_planes==8 && C::vector_packing && C::vector_pack_b && C::compact_pack);'), 'physical_slot_packing_24': (True, 'using C=PackOwnershipConfig<GatherConfig<ReconstructionConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,8,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,true>,RegisterGather::SelectedFour>,true>;static_assert(C::coalesced_pack && C::threads==256 && C::compute_warps==7 && C::a_planes==8 && C::b_planes==8 && C::vector_packing && C::vector_pack_b && C::compact_pack);'), 'reconstruction_reject_materialized': (False, 'using C=ReconstructionConfig<std::tuple_element_t<0,KernelConfigurations>,true>;'), 'reconstruction_reject_raw_b': (False, 'using C=ReconstructionConfig<std::tuple_element_t<18,KernelConfigurations>,true>;'), 'packing_reject_oldlayout': (False, 'using C=PackOwnershipConfig<std::tuple_element_t<18,KernelConfigurations>>;'), 'gather_reject_shared': (False, 'using C=GatherConfig<ReconstructionConfig<TuningConfig<KernelConfig<64,32,16,8,8,1,1,16,1,ExecutionStrategy::CooperativeFused,false,1,true,false,false,2,DataflowPolicy<OperandSource::PackedTransforms,OperandSource::PackedTransforms,PackExecution::SeparateKernel,TailPolicy::Padded,ProducerPolicy::WarpTransform>>,8>,false>,RegisterGather::SelectedFour>;')})
    for name, (accepted, body) in cases.items():
        result = subprocess.run(command, input=prefix + body + "\nstatic_assert(sizeof(C)>0);\n",
                                text=True, capture_output=True)
        if (result.returncode == 0) != accepted:
            raise RuntimeError(f"{name}: expected accepted={accepted}\n{result.stderr}")
        if not accepted and "static assertion failed" not in result.stderr:
            raise RuntimeError(f"{name}: rejected for an unrelated reason\n{result.stderr}")
        print(f"PASS {name}: {'accepted' if accepted else 'rejected'}")


if __name__ == "__main__":
    main()
