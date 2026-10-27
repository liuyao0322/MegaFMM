# SW empirical tuner

**Mandatory user constraint:** BF16 inputs must never be converted to FP16 for intermediate storage or computation. BF16 transforms may be computed in FP32 and rounded back to BF16; accumulation/output remain FP32. Iter378/379 mixed BF16-to-FP16 candidates violated this constraint and are removed. Their measured accuracy and acceleration are withdrawn as accepted outcomes; historical raw records remain for audit. Correction validation completed: full build,612 native configurations x3 fresh direct/Graph checks including BF16 inputs above FP16 range,related CTest2/2 and independent selection audit passed. Both1024 ordinary random-input FP64 gates still fail. See [correction evidence](../trajectory/20261017_081702_iter-380/REPORT.md).

`Algorithm::SWTuned` explicitly tunes **seven-product Winograd SW** configurations.
It never selects a full-AB dense GEMM. `Algorithm::Tuned` retains its existing dense
meaning; `SWPipeline` retains its fixed dispatch and restored 1024 specialization.

```cpp
megafmm::Policy policy;
policy.algorithm = megafmm::Algorithm::SWTuned;
policy.tuning_input_sets = 2;
policy.sw_tuning_graph = false; // true when the intended application uses Graph replay
policy.sw_tuning_candidates = 0; // all legal candidates; positive value: model shortlist
// Caller creates plan, allocates plan->workspace_bytes(), supplies original A/B.
auto plan = megafmm::create_plan(problem, policy);
megafmm::tune(*plan, A, B, nullptr, D, workspace, stream);
megafmm::run(*plan, A, B, nullptr, D, workspace, stream);
```

Supported scope: Hopper sm90, positive dimensions <=32768, contiguous non-transposed
BF16/FP16 input, FP32 output, alpha=1 and beta=0. Input bindings must be 16-byte
aligned. Unsupported problems and insufficient workspace fail explicitly.
Zero-work/scale-only problems keep existing behavior. Tune outside capture; run
allocates no device memory and performs no host synchronization.

## Error reporting and optional historical gate

The latest user objective removes the original precision threshold from candidate
selection. Default `sw_tuning_allow_inaccurate=true` ranks **every finite** sampled
seven-product candidate by full-call latency, even when another candidate passes
the old threshold. No compensation products or BF16-to-FP16 intermediates are used.

Every candidate still records original-input FP64 relative Frobenius error and the
historical comparison `eF_SW <= 3*eF_dense + floor` (1e-3 BF16,1e-4 FP16).
`numerical_pass` means only that historical comparison; `error_reported` indicates
a selected configuration that exceeds it. Neither reports improved accuracy.
Dense is used only to calculate the reference allowance, never as an SW candidate.
Prepared references live only for the tune call. Nonfinite candidates are rejected.
The winner is checked again after sustained execution.

Set `sw_tuning_allow_inaccurate=false` to opt into strict historical behavior:
all tuning sets must pass, and an empty eligible pool throws with
`no_precision_valid_candidate`. Failed retuning invalidates an earlier winner.
This optional mode does not change the default performance objective.

## Candidate space and hardware model

Ten native architecture tuples `(WGMMA N, BK, stages)`:

- Aligned: (32,128,8), (128,64,4), (32,64,4), (32,128,4), (128,128,2), (128,256,2).
- Padded: (16,128,8), (64,128,4), (16,64,4), (64,64,4).

Each tuple explores pack CTA caps {64,128,512} and raster groups {0,2,4}.
Aligned eligibility uses M/N/K jointly and the actual tile divisibility.
Padded candidates align physical half-M to their N, half-N to 64 and half-K to BK;
workspace and physical work are calculated for that configuration. They may also
compete on aligned shapes. The frozen 1024 specialization is an additional candidate
when supported; scalar 1^3 has its seven-SIMT-product candidate.

Native attributes are queried from the actual compiled kernel via
`cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
The initial ranking score combines actual padded FLOPs ratio, output-grid wave
fill, tile reuse and K-step cost. It is a heuristic, not a predicted latency or
proof of peak utilization. The default measures all legal candidates; a positive
budget enables model screening. Configurations remain restricted by workspace.

The candidate space currently retains one compute warpgroup and per-K wait0.
It cannot discover architectures absent from this registry. New lifetime layouts,
warpgroup organizations and asynchronous MMA schedules need separately verified
implementations. The tuner does not promise cuBLAS speedups on every shape.

## Empirical selection

1. Record error for every candidate on every tuning set; reject nonfinite outputs.
   Apply the historical precision threshold only when explicitly requested.
2. Rank eligible SW full invocations using 5 groups of 20 calls with 4 warmups.
   Reverse sweep direction each round to reduce ordering bias.
3. Confirm the four fastest initial candidates (or all if fewer) using 5 groups
   of 100 calls, 20 warmups per group and alternating order.
4. Select the lowest confirmation median, then recheck it after sustained execution.

All timing includes transformation/packing and seven-product compute/recombination.
Graph mode captures a complete invocation per rotating binding, then measures replay.
Tuning input copies model the requested working-set size, but contain copies of the
supplied values, not independent fresh random samples. No input values survive tune.

The choice is the best confirmed configuration within this search and measurement
budget, not a proven global optimum. Short-ranking noise can exclude a candidate
from the four finalists. Candidate reports make this distinction observable.
There is no cuBLAS-relative 3% elimination rule inside SW selection.

The chosen configuration and full measurement records live in the Plan. They are
not persisted across processes. Device identity is checked before using a choice.
Workspaces and output bindings remain caller owned; independent streams require
independent workspaces. Rebinding aligned input/output storage is supported.

## Benchmark and evidence

```bash
bash scripts/on_gpu.sh ./build-sw-tuner/megafmm_bench \
  --algorithm sw-tuned --shapes '512,512,512;514,770,258' \
  --dtype both --rotate-inputs 2 --warmup 20 --groups 5 --trials 100 \
  --correct-trials 3 --output results/sw_tuner.csv
# Add --graph for tuning and benchmarking in Graph mode.
# Use --algorithm sw-tuned-strict for the optional historical precision gate.
# sw-tuned-diagnostic remains an alias for error-reporting performance selection.
```

The benchmark retains original FP64 errors, historical `numerical_pass` and fresh
input trials. New `precision_required` and `accepted` columns distinguish the
requested policy from the historical verdict. Default SW mode returns success
for finite fresh-input results regardless of the old threshold.
`<output>.sw_tuning.csv` additionally stores every candidate configuration, resource
metadata, precision errors/limits per set, all ranking samples and confirmation
samples. A failed strict tune also writes its rejection evidence before propagating
the error. The public `Plan::sw_tuning_results()` returns these records without GPU
work; `lt_algorithm_config()` identifies SW status, mode and selected configuration.

Tests `sw_tuner_candidates` exercise every eligible native configuration on aligned,
padded, rectangular and longer-K problems in both dtypes with fresh exact-reference
inputs and Graph replay. `sw_tuner_selection` covers strict selection, explicit tune,
failed retuning, error reporting, exhaustive finite-candidate ranking, rotating sets and both execution modes.
