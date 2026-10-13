cmake_minimum_required(VERSION 3.24)
project(MegaFMM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)
find_package(CUDAToolkit 12.4 REQUIRED)
set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS 3.5.1 checkout")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS missing: run scripts/setup_dependencies.sh on the login node")
endif()
add_library(megafmm STATIC solution/plan.cpp solution/pack_winograd.cu
  solution/combine_fp32.cu solution/gemm_hopper.cu solution/gemm_batched.cu
  solution/gemm_batched_k64.cu solution/gemm_fused_leaf.cu solution/gemm_packed_leaf.cu
  solution/megakernel_hopper.cu solution/small_mma.cu solution/small_mma_fast.cu
  solution/small_product_mma.cu solution/small_product_mma_fast.cu
  solution/small_mma_16x8.cu solution/small_mma_16x8_vec4.cu
  solution/small_product_cluster.cu solution/small_mma_16x8_epilogue.cu
  solution/small_mma_16x8_identity.cu solution/small_mma_16x8_prefetch.cu
  solution/small_mma_cpasync.cu solution/small_mma_stage.cu
  solution/small_mma_8x8_prefetch.cu solution/small_mma_direct_register.cu
  solution/small_mma_coalesced_register.cu
  solution/small_mma_k128_specialized.cu
  solution/small_mma_cooperative.cu solution/megakernel_wgmma_pipeline.cu
  solution/megakernel_wgmma_tma_alias.cu)
target_include_directories(megafmm PUBLIC include PRIVATE solution "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
target_link_libraries(megafmm PUBLIC CUDA::cudart CUDA::cublasLt CUDA::cublas CUDA::cuda_driver)
target_compile_options(megafmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;-Xptxas=-v>)
add_executable(megafmm_bench bench/benchmark.cu)
target_link_libraries(megafmm_bench PRIVATE megafmm)
target_compile_options(megafmm_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE megafmm)
target_compile_options(test_correctness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)
enable_testing()
add_test(NAME gpu_correctness COMMAND test_correctness)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME exact_winograd_dag COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dag.py")
  add_test(NAME stage_m2_model COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stage_m2.py")
endif()
