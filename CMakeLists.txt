cmake_minimum_required(VERSION 3.24)
project(MegaFMM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)
find_package(CUDAToolkit 12.4 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MEGAFMM_INTERNAL_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/solution/core"
  "${CMAKE_CURRENT_SOURCE_DIR}/solution/sw/fp32"
  "${CMAKE_CURRENT_SOURCE_DIR}/solution/sw/hopper")
set(MEGAFMM_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS headers for Hopper primitives")
if(NOT EXISTS "${MEGAFMM_CUTLASS_DIR}/include/cute/tensor.hpp")
  message(FATAL_ERROR "Hopper SW needs CUTLASS 3.5.1 headers; set MEGAFMM_CUTLASS_DIR")
endif()

add_library(megafmm STATIC
  solution/core/plan.cpp
  solution/core/scale_output.cu
  solution/core/tuning_check.cu
  solution/sw/sw_pipeline.cu
  solution/sw/hopper/sw_hopper.cu
  solution/sw/fp32/sw_fp32.cu
  solution/sw/fp32/sw_fp32_batch.cu)
set_target_properties(megafmm PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(MEGAFMM_FP32_CONFIG_HEADER "" CACHE FILEPATH "Optional experimental FP32 template registry")
if(MEGAFMM_FP32_CONFIG_HEADER)
  get_filename_component(MEGAFMM_FP32_CONFIG_HEADER "${MEGAFMM_FP32_CONFIG_HEADER}" ABSOLUTE)
  if(NOT EXISTS "${MEGAFMM_FP32_CONFIG_HEADER}")
    message(FATAL_ERROR "FP32 template registry does not exist: ${MEGAFMM_FP32_CONFIG_HEADER}")
  endif()
  target_compile_definitions(megafmm PUBLIC MEGAFMM_FP32_CONFIG_HEADER="${MEGAFMM_FP32_CONFIG_HEADER}")
endif()
target_compile_definitions(megafmm PRIVATE CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
target_include_directories(megafmm PUBLIC include PRIVATE ${MEGAFMM_INTERNAL_INCLUDE_DIRS} "${MEGAFMM_CUTLASS_DIR}/include")
target_link_libraries(megafmm PUBLIC CUDA::cudart CUDA::cublasLt CUDA::cublas CUDA::cuda_driver)
target_compile_options(megafmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)

add_library(megafmm_suite SHARED bench/suite/native_bridge.cpp)
target_link_libraries(megafmm_suite PRIVATE megafmm)
set_target_properties(megafmm_suite PROPERTIES OUTPUT_NAME megafmm_suite)

function(megafmm_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${MEGAFMM_INTERNAL_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE megafmm)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)
endfunction()
megafmm_executable(megafmm_bench bench/benchmark.cu)
megafmm_executable(test_correctness tests/test_correctness.cu)
megafmm_executable(test_tuning_precision tests/test_tuning_precision.cu)
megafmm_executable(test_sw_paths tests/test_sw_paths.cu)

megafmm_executable(test_tensorcore_configs tests/test_tensorcore_configs.cu)

enable_testing()
add_test(NAME tensorcore_configurations COMMAND test_tensorcore_configs)
add_test(NAME gpu_correctness COMMAND test_correctness)
add_test(NAME tuned_correctness COMMAND test_correctness --tuned)
add_test(NAME tuning_precision_guard COMMAND test_tuning_precision)
add_test(NAME sw_general_paths COMMAND test_sw_paths)
add_test(NAME exact_winograd_dag COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dag.py")
add_test(NAME fp32_policy_types COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_fp32_policy_types.py"
  --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUDAToolkit_INCLUDE_DIRS}")
add_test(NAME application_suite_host COMMAND "${Python3_EXECUTABLE}" -m unittest discover
  -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_suite_*.py")
set_tests_properties(application_suite_host PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
