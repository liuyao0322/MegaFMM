# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|

| 1 | Vectorize shared DAG packing and FP32 combine | 0.8186 | 5.048809 | improved |

| 2 | Stream full products through three accumulator tiles | 0.0619 | 13.865779 | regression |

| 3 | Remove unused leaf C stages and multicast B tiles | 0.8186 | 5.137198 | no-change |

| 4 | Overlap full-tile transforms with WGMMA in two shared stages | 0.0702 | 12.236769 | improved |

| 5 | Replace the unresolved Pingpong control with ordinary warp specialization | 0.5183 | 9.565941 | no-change |

Each row pools only that iteration's selected experimental family and shapes.
These arithmetic means are not comparable across iterations with different
families or scopes; use matched shape/dtype/mode rows for decisions. The final
retained tree includes the fast implementations from earlier iterations.

| 6 | Fuse small-shape A and B DAG packing into one launch | 0.2474 | 0.108239 | no-change |

| 7 | Batch seven Winograd products through one tuned Lt operation | 0.5311 | 0.075244 | improved |

| 8 | Use small-tile cp.async Tensor Core batched leaves | 0.4420 | 0.121435 | regression |

| 9 | Fuse Winograd into independent warp-level mma.sync owners | 0.2690 | 0.403353 | regression |

| 10 | Specialize complete small tiles and packed final conversions | 0.3431 | 0.399807 | improved |

| 11 | Parallelize seven complete products across warps and fix epilogue rounding | 0.2446 | 1.033407 | regression |

| 12 | Remove complete-tile checks from parallel product warps | 0.3034 | 0.914679 | regression |

| 13 | Reduce fused output owners to native16x8 Tensor Core tiles | 0.2953 | 0.681080 | regression |

| 14 | 16x8 B fill across a full warp | 0.3673 | 0.678265 | screened; numerical gate failed; structural validation pending |

| 15 | Spread seven complete products over eight-CTA Hopper clusters | 0.2237 | 1.616020 | screened; numerical gate failed; cluster validation pending |

## Iterations

## Baseline — full V1 / tuned cuBLASLt (2026-10-13)

- CUDA12.4.99, CUTLASS3.5.1 f7b19de32c5d1f3cedfc735c2849f12b537522ee, assigned H10080GB, sm90a.
- Compiled=True. Exact DAG3 tests and complete GPU structural/API suite PASS, including all initial variants and aligned alpha/beta/C0=D.
- Full benchmark: launch, warm cache,20 warmups,100 calls/group,5 alternating groups,3 fresh FP64-reference rounds per row.
- Numerical certification=False for all forced V1 rows. Exit2 explicitly represents frozen error gate failure, not a runtime or algorithm identity failure. Auto remains Dense; optimization experiments remain uncertified until the gate is met.
- Transform quantization loss is intrinsic to the stated final low-precision operand cast: independent FP64-leaf CPU diagnostic also fails floors. No tolerance relaxation.

| M=N=K | dtype | Dense median ms | V1 median ms | speedup | SW eF | gate |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | bf16 | 0.196908 | 0.356738 | 0.5520 | 0.0060269 | 0.00101207 |
| 4096 | fp16 | 0.206495 | 0.373236 | 0.5533 | 0.000753133 | 0.00011463 |
| 8192 | bf16 | 1.677764 | 1.962595 | 0.8549 | 0.00602339 | 0.00102565 |
| 8192 | fp16 | 1.787713 | 2.037782 | 0.8773 | 0.000753411 | 0.000129068 |
| 16384 | bf16 | 12.661223 | 13.059917 | 0.9695 | 0.00602276 | 0.00105372 |
| 16384 | fp16 | 13.259946 | 13.621885 | 0.9734 | 0.000753079 | 0.000157995 |

- Resource probe: megakernel64x32 M1=150 registers,42KiB SMEM,0 spill;64x64 M1=254 registers,56KiB,0 spill. Stock Hopper variants use168 static registers. An early non-NDEBUG probe emitted512/1024-byte spill stores/loads; the measured Release objects have zero stack/spills and no LDL/STL, as verified in hopper_release_resources.txt. The probe is not evidence of spills in benchmark binaries.
- Raw baseline snapshot: trajectory/20261013_114800_baseline/. Next: ncu baseline once, then optimize dominant full-call cost.

### Iter 1 — Vectorize shared DAG packing and FP32 combine

- **Hypothesis:** Nsight Systems attributes 25 percent of V1 GPU kernel time to packing and combine; 16-byte transactions reduce load/store issue pressure and index work
- **Changes:** Eight BF16/FP16 elements per packing thread and four FP32 outputs per combine thread, preserving exact FP32 S/T/U order and scalar combine for unaligned output views
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 6 rows; arithmetic mean of row median latency=5.048809 ms; mean speedup=0.8186x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 4096,4096,4096 | bf16 | winograd_lt_v1 v0 | launch | 0.195936 | 0.335667 | 0.5837 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_lt_v1 v0 | launch | 0.206574 | 0.345277 | 0.5983 | 0.000753133 | 0 |
| 8192,8192,8192 | bf16 | winograd_lt_v1 v0 | launch | 1.543662 | 2.016856 | 0.7654 | 0.00602247 | 0 |
| 8192,8192,8192 | fp16 | winograd_lt_v1 v0 | launch | 1.728507 | 1.984926 | 0.8708 | 0.000753411 | 0 |
| 16384,16384,16384 | bf16 | winograd_lt_v1 v0 | launch | 13.180775 | 12.643925 | 1.0425 | 0.00602276 | 0 |
| 16384,16384,16384 | fp16 | winograd_lt_v1 v0 | launch | 13.628090 | 12.966200 | 1.0510 | 0.000753079 | 0 |

- **Analysis:** All measured outputs finite and transform error unchanged. Larger shapes show potential latency benefit but clocks vary, requiring full repeated gate run. Correction to initial resource note: measured Release CUTLASS object has zero stack/local and no LDL/STL; spills came from non-NDEBUG probe only, see hopper_release_resources.txt. Matched-baseline geometric latency improvement=1.0373x (6 rows); certification remains independent.
- **Next:** Check vector memory and alias paths; then test full-product streaming megakernel and Hopper tile/epilogue alternatives
- **Archive:** trajectory/20261013_115411_iter-1/

### Iter 2 — Stream full products through three accumulator tiles

- **Hypothesis:** M1 spends tens of milliseconds in small tiles and scalar seven-slot traffic; selective vector loads and a wider N128 tile can reduce instruction work and shared stores
- **Changes:** M3 variant2 64x128x32 with 12KiB shared storage, three P/U live tiles, selective uint4 raw-input transport and original full-K product/U ordering; retain M1 controls
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 4 rows; arithmetic mean of row median latency=13.865779 ms; mean speedup=0.0619x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 4096,4096,4096 | bf16 | megakernel_m3 v2 | launch | 0.194893 | 3.073949 | 0.0634 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_m3 v2 | launch | 0.203462 | 3.014240 | 0.0675 | 0.000753133 | 0 |
| 8192,8192,8192 | bf16 | megakernel_m3 v2 | launch | 1.407350 | 24.564484 | 0.0573 | 0.00602247 | 0 |
| 8192,8192,8192 | fp16 | megakernel_m3 v2 | launch | 1.472835 | 24.810445 | 0.0594 | 0.000753411 | 0 |

- **Analysis:** Compiled248 registers with zero stack/spill. At8192 latency fell from M1 about34.5ms to24.6-24.8ms but is still an order of magnitude slower than V1 and tuned dense. Full product rescans and synchronization remain expensive. All outputs finite, same mixed-precision gate failures; no automatic dispatch enablement.
- **Next:** Test Hopper no-source epilogue storage reduction and cluster multicast; retain V1 iter1 as best overall route
- **Archive:** trajectory/20261013_115956_iter-2/

### Iter 3 — Remove unused leaf C stages and multicast B tiles

- **Hypothesis:** Leaf GEMMs dominate V1 runtime; a source-free beta-zero epilogue frees shared memory for more mainloop stages, while cluster multicast reuses B tiles
- **Changes:** Add Hopper variants 3 and 4 with compile-time C=void ScaledAcc and variant-4 B multicast; retain controls 0-2 and vector transforms
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 6 rows; arithmetic mean of row median latency=5.137198 ms; mean speedup=0.8186x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 4096,4096,4096 | bf16 | winograd_hopper_v2 v4 | launch | 0.194674 | 0.318883 | 0.6105 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_hopper_v2 v4 | launch | 0.205310 | 0.336354 | 0.6104 | 0.000753133 | 0 |
| 8192,8192,8192 | bf16 | winograd_hopper_v2 v4 | launch | 1.555890 | 1.945867 | 0.7996 | 0.00602247 | 0 |
| 8192,8192,8192 | fp16 | winograd_hopper_v2 v4 | launch | 1.670877 | 2.024960 | 0.8251 | 0.000753411 | 0 |
| 16384,16384,16384 | bf16 | winograd_hopper_v2 v4 | launch | 13.372861 | 12.821434 | 1.0430 | 0.00602276 | 0 |
| 16384,16384,16384 | fp16 | winograd_hopper_v2 v4 | launch | 13.685713 | 13.375691 | 1.0232 | 0.000753079 | 0 |

- **Analysis:** Variant 4 measured 0.319/0.336 ms at 4096, 1.946/2.025 ms at 8192, and 12.821/13.376 ms at 16384 for BF16/FP16. The smaller BF16 case improves over the prior Hopper control, while large-size timing varies with unlocked clocks. All outputs are finite but the unchanged frozen numerical gates fail. This is an experimental leaf improvement, not certified SW acceleration.
- **Next:** Compare Hopper controls and validate every candidate, then investigate M3 full-tile code generation and overlapping SIMT transforms with WGMMA
- **Archive:** trajectory/20261013_120157_iter-3/

### Iter 4 — Overlap full-tile transforms with WGMMA in two shared stages

- **Hypothesis:** M3 generic stores and scalar fallbacks inflate instructions; a specialized full-tile path and independent shared stages can reduce address work and hide transforms behind WGMMA
- **Changes:** Add mega variant3 with aligned vector loads, explicit st.shared.v4.b32, beta-zero epilogue and 24KiB double buffer; preserve variant2 general fallback and fixed P/U ordering
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 4 rows; arithmetic mean of row median latency=12.236769 ms; mean speedup=0.0702x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 4096,4096,4096 | bf16 | megakernel_m3_pipeline v3 | launch | 0.194934 | 2.677334 | 0.0728 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_m3_pipeline v3 | launch | 0.204006 | 2.693648 | 0.0757 | 0.000753133 | 0 |
| 8192,8192,8192 | bf16 | megakernel_m3_pipeline v3 | launch | 1.411360 | 21.686758 | 0.0651 | 0.00602247 | 0 |
| 8192,8192,8192 | fp16 | megakernel_m3_pipeline v3 | launch | 1.474432 | 21.889337 | 0.0674 | 0.000753411 | 0 |

- **Analysis:** Compiled233 registers/thread with no stack or spills, versus248 in M3. At8192 BF16/FP16 latency fell from24.56/24.81 to21.69/21.89ms, about12percent lower, and4096 fell to2.68/2.69ms. Outputs remain finite and error unchanged. This improves the fused experimental family but remains much slower than V1; it cannot meet the unchanged numerical gate.
- **Next:** Validate asynchronous stage ownership and all API paths; isolate the stock Pingpong synchronization report, then obtain full final launch/Graph/rotating measurements
- **Archive:** trajectory/20261013_121033_iter-4/

### Iter 5 — Replace the unresolved Pingpong control with ordinary warp specialization

- **Hypothesis:** A one-consumer warp-specialized control avoids the Pingpong math-order barrier diagnostics without changing the pinned dependency, operand policy or tile
- **Changes:** Switch Hopper variant2 at64x128x64 to KernelTmaWarpSpecialized with matching epilogue; retain0/1/3/4 and all previous optimized routes
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 6 rows; arithmetic mean of row median latency=9.565941 ms; mean speedup=0.5183x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 4096,4096,4096 | bf16 | winograd_hopper_v2 v2 | launch | 0.195198 | 0.422069 | 0.4625 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_hopper_v2 v2 | launch | 0.206389 | 0.427397 | 0.4829 | 0.000753133 | 0 |
| 8192,8192,8192 | bf16 | winograd_hopper_v2 v2 | launch | 1.547059 | 2.666178 | 0.5803 | 0.00602247 | 0 |
| 8192,8192,8192 | fp16 | winograd_hopper_v2 v2 | launch | 1.648163 | 2.762986 | 0.5965 | 0.000753411 | 0 |
| 16384,16384,16384 | bf16 | winograd_hopper_v2 v2 | launch | 12.355282 | 25.226591 | 0.4898 | 0.00602276 | 0 |
| 16384,16384,16384 | fp16 | winograd_hopper_v2 v2 | launch | 12.888681 | 25.890427 | 0.4978 | 0.000753079 | 0 |

- **Analysis:** Compiles with zero spills. Full-call control latency is0.422/0.427ms at4096,2.666/2.763ms at8192,25.227/25.890ms at16384 BF16/FP16. This is a validation repair/control rather than a speed winner; keep vector V1 and Hopper3/4 as fast experimental routes. Frozen numerical gates still fail with finite outputs; independent sanitizer confirmation follows this commit.
- **Next:** Run complete structural and sanitizer checks, then full final timing with100calls5groups3fresh references, Graph and rotating inputs
- **Archive:** trajectory/20261013_121327_iter-5/

## Final reassessment and retained implementation (2026-10-13)

The fixed one-cast Winograd policy is not certified. This is a numerical
feasibility barrier for the current policy, rather than evidence of a minimum
possible GPU latency. The independent `tests/mixed_precision_floor.py` witness
uses original already-rounded inputs and exact leaf products: BF16 eF is
0.004871312 versus floor0.001, and FP16 eF is0.000803082 versus floor0.0001.
The FP32 product/U diagnostic also fails, with Dense eF0 and8.92e-9 respectively.
Tensoring these2x2 coefficient matrices with an identity preserves relative
error and constructs aligned sizes; this is a mathematical lifting argument,
not a claimed GPU run of the lifted witness. Independent GPU uniform/range
stress checks demonstrate the same limitation in actual candidates.

Changing transport, tiles, stages, occupancy or product lifetime cannot restore
information removed by that prescribed cast. The current optimization cycle
therefore closes with no automatically accepted SW shape. Compensation,
scaling, restricted input contracts or different operand precision require a
separate numerical policy and independently timed/validated implementation.
No tolerance was relaxed, no random-pattern detector was introduced, and
finite output alone was never counted as numerical certification.

The reassessment reviewed per-shape logs and the once-only Nsight Systems
capture because the session's one ncu probe failed. Four distinct execution
rewrites were measured and committed: iter1 vectorized S/T and U transport;
iter2 three-accumulator M3 full-product scheduling; iter3 source-free Hopper
epilogue and cluster B multicast; iter4 full-tile shared vector stores and
asynchronous two-stage transform/WGMMA pipeline. Iter5 repaired the diagnostic
control. The [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.4.1/hopper-tuning-guide/index.html)
and [pinned CUTLASS producer/consumer mainloop](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)
support the data-transfer/compute overlap direction; source inspection and
compiled instruction order were used alongside full-call timing, rather than
assuming that a stage-count change alone produces a gain. The source-free
epilogue comparison also changes delayed-store behavior and mainloop stages;
it is not an isolated epilogue-arithmetic experiment.

The final retained source is d3cff91's solution tree. It already includes the
unchanged iter1 vector V1, iter3 Hopper3/4 alternatives, iter4's fastest measured
M3 variant3, and iter5's validated ordinary warp-specialized control. No later
iteration overwrote those fast candidates, so there is no source restoration.
The default Auto route remains tuned Dense because the accepted SW set is
empty. Per-family retention is intentional: V1 has the lower measured large
shape latency, while Hopper4 is competitive at smaller shapes. Summary means
over different families are not a ranking of complete implementations.

Final validation passed3 independent DAG tests, the full structural/API suite,
full-suite memcheck, and all-candidate racecheck/synccheck. The strict matrix
contains300 candidate/pattern/trial/dtype checks:120 frozen-gate failures,
zero nonfinite outputs. Each dtype has150 checks and60 failures. Failures are
uniform and in-range amplitude stress; the high FP16 range cases deliberately
use Dense fallback and cannot certify SW. Logs are `results/final_*.txt`.

Final ordinary launch examples,20warmups/100calls/5groups/3fresh FP64 references:

| M=N=K | dtype | tuned Dense median ms | vector V1 median ms | Dense / V1 | SW eF | maximum trial gate |
|---:|---|---:|---:|---:|---:|---:|
|8192|BF16|1.717472|1.935383|0.887407|0.006023394|0.001025652|
|8192|FP16|1.771971|1.984888|0.892731|0.000753411|0.000129068|
|16384|BF16|12.842401|12.944546|0.992109|0.006022762|0.001053692|
|16384|FP16|13.302281|13.472572|0.987360|0.000753079|0.000157995|

The short iter1 16384 FP16 ratio1.051 did not persist at the full gate.
Neither neighboring large shape performance nor fixed numerical acceptance
was achieved. Graph and4-address-pair rotation are reported separately in
`results/REPORT.md`; every invocation includes fresh dependent transforms.
The M3 pipeline retest gives8192 BF16/FP16 21.670815/21.808022ms, approximately
12% below iter2's24.564484/24.810445ms, but far slower than tuned Dense. Static
SASS sites fall8152→3928 and registers248→233, with zero spills; this evidence
does not measure dynamic utilization or overlap duration. See
`results/mega_pipeline_resources.txt`.

All timing uses unlocked GPU clocks. Group p10/p90 and CV describe five group
means, not invocation tails or confidence intervals; before/after telemetry
does not establish within-group clock stability. Final source commits and
exit codes are in `results/final_runs.csv`. There are94 final rows:88 actual SW
rows all fail the frozen gate,6 Dense fallback rows pass, and no unchecked or
nonfinite rows. `results/final_manifest.json` records source/binary/CSV hashes,
device identity, protocol and independent evidence counts. Exit2 preserves the numerical
failure; it is not a runtime failure or a passing score.

This closure does not claim that every performance research direction is
exhausted. M2, broader V3/V4, product-aware batch/scheduler, tail peeling and
two levels remain unimplemented. The primary-source SubCuber audit in
`results/subcuber_compatibility.json` establishes that an unmodified author
runner is not the same BF16/FP16-input FP32-output dynamic-input comparator;
no author build, H100 run or performance speedup is reported. These open
experiments have not been silently marked complete in PLAN.

### Iter 6 — Fuse small-shape A and B DAG packing into one launch

- **Hypothesis:** Small matrix transforms are short enough that merging their independent launches can lower full-call fixed cost
- **Changes:** Add combined vector packing for dimensions<=4096; retain separate packing for larger shapes and original FP32 DAG/cast order
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.108239 ms; mean speedup=0.2474x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_lt_v1 v0 | launch | 0.005012 | 0.038908 | 0.1288 | 0.00611792 | 0 |
| 128,128,128 | fp16 | winograd_lt_v1 v0 | launch | 0.004880 | 0.038596 | 0.1264 | 0.000774794 | 0 |
| 256,256,256 | bf16 | winograd_lt_v1 v0 | launch | 0.005249 | 0.039128 | 0.1341 | 0.00605431 | 0 |
| 256,256,256 | fp16 | winograd_lt_v1 v0 | launch | 0.004967 | 0.038613 | 0.1286 | 0.000755322 | 0 |
| 512,512,512 | bf16 | winograd_lt_v1 v0 | launch | 0.006033 | 0.041272 | 0.1462 | 0.00604926 | 0 |
| 512,512,512 | fp16 | winograd_lt_v1 v0 | launch | 0.005923 | 0.039282 | 0.1508 | 0.000756941 | 0 |
| 1024,1024,1024 | bf16 | winograd_lt_v1 v0 | launch | 0.012274 | 0.050590 | 0.2426 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | winograd_lt_v1 v0 | launch | 0.009797 | 0.049439 | 0.1982 | 0.000754626 | 0 |
| 2048,2048,2048 | bf16 | winograd_lt_v1 v0 | launch | 0.030959 | 0.125291 | 0.2471 | 0.00602495 | 0 |
| 2048,2048,2048 | fp16 | winograd_lt_v1 v0 | launch | 0.031651 | 0.107609 | 0.2941 | 0.00075359 | 0 |
| 3072,3072,3072 | bf16 | winograd_lt_v1 v0 | launch | 0.085878 | 0.224549 | 0.3824 | 0.00602605 | 0 |
| 3072,3072,3072 | fp16 | winograd_lt_v1 v0 | launch | 0.089061 | 0.231452 | 0.3848 | 0.000753777 | 0 |
| 4096,4096,4096 | bf16 | winograd_lt_v1 v0 | launch | 0.200778 | 0.351257 | 0.5716 | 0.0060269 | 0 |
| 4096,4096,4096 | fp16 | winograd_lt_v1 v0 | launch | 0.208364 | 0.371081 | 0.5615 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_lt_v1 v0 | launch | 0.010752 | 0.051551 | 0.2086 | 0.00604051 | 0 |
| 1024,512,1024 | fp16 | winograd_lt_v1 v0 | launch | 0.010813 | 0.052626 | 0.2055 | 0.000755532 | 0 |
| 512,1024,1024 | bf16 | winograd_lt_v1 v0 | launch | 0.008227 | 0.048318 | 0.1703 | 0.00602806 | 0 |
| 512,1024,1024 | fp16 | winograd_lt_v1 v0 | launch | 0.008392 | 0.048740 | 0.1722 | 0.000756687 | 0 |

- **Analysis:** Matched small baseline geometric latency gain=1.0037x over18 rows. Single CTA-independent A/B packing saves one launch with unchanged final casts and workspace. Remaining seven leaves dominate small-call fixed cost; all frozen numerical gates remain visible. No claim of certified acceleration.
- **Next:** Replace seven launches with uniform operand batches and investigate warp-level fused small tiles
- **Archive:** trajectory/20261013_131015_iter-6/

### Iter 7 — Batch seven Winograd products through one tuned Lt operation

- **Hypothesis:** One strided-batched Tensor Core GEMM removes repeated leaf launch and metadata costs at small shapes
- **Changes:** Add Ltvariant1 with product-ordered7inputslots, one combined packing, batch-aware Lt32heuristic pools/tuning and originalFP32combine; retain Lt0 and add API/fallback/alias/rebind checks
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.075244 ms; mean speedup=0.5311x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_lt_batched v1 | launch | 0.003940 | 0.009671 | 0.4074 | 0.00611792 | 0 |
| 128,128,128 | fp16 | winograd_lt_batched v1 | launch | 0.003903 | 0.009982 | 0.3910 | 0.000774794 | 0 |
| 256,256,256 | bf16 | winograd_lt_batched v1 | launch | 0.007721 | 0.014954 | 0.5163 | 0.00605431 | 0 |
| 256,256,256 | fp16 | winograd_lt_batched v1 | launch | 0.009611 | 0.018605 | 0.5166 | 0.000755322 | 0 |
| 512,512,512 | bf16 | winograd_lt_batched v1 | launch | 0.006110 | 0.012656 | 0.4828 | 0.00604926 | 0 |
| 512,512,512 | fp16 | winograd_lt_batched v1 | launch | 0.005892 | 0.011831 | 0.4980 | 0.000756941 | 0 |
| 1024,1024,1024 | bf16 | winograd_lt_batched v1 | launch | 0.012496 | 0.019830 | 0.6301 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | winograd_lt_batched v1 | launch | 0.009881 | 0.019638 | 0.5032 | 0.000754626 | 0 |
| 2048,2048,2048 | bf16 | winograd_lt_batched v1 | launch | 0.031428 | 0.073120 | 0.4298 | 0.00602495 | 0 |
| 2048,2048,2048 | fp16 | winograd_lt_batched v1 | launch | 0.032221 | 0.069055 | 0.4666 | 0.00075359 | 0 |
| 3072,3072,3072 | bf16 | winograd_lt_batched v1 | launch | 0.089019 | 0.178166 | 0.4996 | 0.00602605 | 0 |
| 3072,3072,3072 | fp16 | winograd_lt_batched v1 | launch | 0.095511 | 0.165269 | 0.5779 | 0.000753777 | 0 |
| 4096,4096,4096 | bf16 | winograd_lt_batched v1 | launch | 0.207565 | 0.357885 | 0.5800 | 0.0060269 | 0 |
| 4096,4096,4096 | fp16 | winograd_lt_batched v1 | launch | 0.214845 | 0.329662 | 0.6517 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_lt_batched v1 | launch | 0.011281 | 0.016905 | 0.6673 | 0.00604051 | 0 |
| 1024,512,1024 | fp16 | winograd_lt_batched v1 | launch | 0.010179 | 0.016140 | 0.6306 | 0.000755532 | 0 |
| 512,1024,1024 | bf16 | winograd_lt_batched v1 | launch | 0.008428 | 0.015489 | 0.5441 | 0.00602806 | 0 |
| 512,1024,1024 | fp16 | winograd_lt_batched v1 | launch | 0.008810 | 0.015528 | 0.5673 | 0.000756687 | 0 |

- **Analysis:** Matched original small V1 geometric latency gain=2.2975x over18 rows. 128 bf16=9.67us (Dense/winograd_lt_batched 0.407x); 128 fp16=9.98us (Dense/winograd_lt_batched 0.391x); 512 bf16=12.66us (Dense/winograd_lt_batched 0.483x); 512 fp16=11.83us (Dense/winograd_lt_batched 0.498x); 1024 bf16=19.83us (Dense/winograd_lt_batched 0.630x); 1024 fp16=19.64us (Dense/winograd_lt_batched 0.503x). Packing raw quadrants into7uniformslots reduces leaf APIcalls7to1;14n²matrix workspace includes newcopies. All outputsfinite, unchanged frozen gatesfail; automaticdispatch remainsDense.
- **Next:** Validate batched API and compare custom small-tile cp.async batched leaves, then per-warp fused mma.sync
- **Archive:** trajectory/20261013_131547_iter-7/

### Iter 8 — Use small-tile cp.async Tensor Core batched leaves

- **Hypothesis:** Three-stage Sm80 batched tiles reduce pipeline fill and descriptor overhead for small Winograd products
- **Changes:** Add compact uniform-seven-slot CUTLASS GemmBatched with CTA64x64x32 and32x64x32 as Hoppervariants5/6; benchmark5; preserve Lt and TMA routes and fixed DAG
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.121435 ms; mean speedup=0.4420x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_sm80_batched v5 | launch | 0.003940 | 0.009769 | 0.4033 | 0.00611792 | 0 |
| 128,128,128 | fp16 | winograd_sm80_batched v5 | launch | 0.003916 | 0.010083 | 0.3884 | 0.000774794 | 0 |
| 256,256,256 | bf16 | winograd_sm80_batched v5 | launch | 0.005212 | 0.010993 | 0.4741 | 0.00605431 | 0 |
| 256,256,256 | fp16 | winograd_sm80_batched v5 | launch | 0.004591 | 0.011013 | 0.4169 | 0.000755322 | 0 |
| 512,512,512 | bf16 | winograd_sm80_batched v5 | launch | 0.005875 | 0.011872 | 0.4948 | 0.00604926 | 0 |
| 512,512,512 | fp16 | winograd_sm80_batched v5 | launch | 0.005883 | 0.011805 | 0.4984 | 0.000756941 | 0 |
| 1024,1024,1024 | bf16 | winograd_sm80_batched v5 | launch | 0.012445 | 0.021246 | 0.5857 | 0.0060152 | 0 |
| 1024,1024,1024 | fp16 | winograd_sm80_batched v5 | launch | 0.009954 | 0.021149 | 0.4707 | 0.000754626 | 0 |
| 2048,2048,2048 | bf16 | winograd_sm80_batched v5 | launch | 0.031236 | 0.099783 | 0.3130 | 0.00602495 | 0 |
| 2048,2048,2048 | fp16 | winograd_sm80_batched v5 | launch | 0.031838 | 0.100167 | 0.3178 | 0.00075359 | 0 |
| 3072,3072,3072 | bf16 | winograd_sm80_batched v5 | launch | 0.087730 | 0.278148 | 0.3154 | 0.00602605 | 0 |
| 3072,3072,3072 | fp16 | winograd_sm80_batched v5 | launch | 0.091606 | 0.281168 | 0.3258 | 0.000753777 | 0 |
| 4096,4096,4096 | bf16 | winograd_sm80_batched v5 | launch | 0.201291 | 0.612976 | 0.3284 | 0.0060269 | 0 |
| 4096,4096,4096 | fp16 | winograd_sm80_batched v5 | launch | 0.207141 | 0.638997 | 0.3242 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_sm80_batched v5 | launch | 0.011003 | 0.017976 | 0.6121 | 0.00604051 | 0 |
| 1024,512,1024 | fp16 | winograd_sm80_batched v5 | launch | 0.010106 | 0.016704 | 0.6050 | 0.000755532 | 0 |
| 512,1024,1024 | bf16 | winograd_sm80_batched v5 | launch | 0.008641 | 0.016213 | 0.5330 | 0.00602806 | 0 |
| 512,1024,1024 | fp16 | winograd_sm80_batched v5 | launch | 0.008653 | 0.015758 | 0.5491 | 0.000756687 | 0 |

- **Analysis:** Matched Ltbatch iter7 geometric latency gain=0.8810x over18 rows. 128 bf16=9.77us (Dense/SW 0.403x); 128 fp16=10.08us (Dense/SW 0.388x); 256 bf16=10.99us (Dense/SW 0.474x); 256 fp16=11.01us (Dense/SW 0.417x); 512 bf16=11.87us (Dense/SW 0.495x); 512 fp16=11.80us (Dense/SW 0.498x); 1024 bf16=21.25us (Dense/SW 0.586x); 1024 fp16=21.15us (Dense/SW 0.471x). Variant5 has no leaf scratch or TMA descriptors; full18rows have finite outputs but fail frozen numerical gates. Relative gain at256 partly reflects earlier noisy Lt row; repeat final before interpreting.
- **Next:** Validate both batched tiles and integrate per-warp fused mma.sync to remove all intermediate global buffers
- **Archive:** trajectory/20261013_132150_iter-8/

### Iter 9 — Fuse Winograd into independent warp-level mma.sync owners

- **Hypothesis:** Small independent output owners remove all global matrix intermediates and increase CTA coverage on tiny matrices
- **Changes:** Add Mega4/5/6 with1/2/4ownerwarps of16x16x16,7FP32P fragments,7/14/28KiBprivate shared andzero matrixworkspace; benchmark4 with predeclared quick20calls/5groups/1freshcheck
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.403353 ms; mean speedup=0.2690x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma v4 | launch | 0.004166 | 0.006862 | 0.6071 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma v4 | launch | 0.004128 | 0.007091 | 0.5821 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma v4 | launch | 0.005250 | 0.011155 | 0.4706 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma v4 | launch | 0.004835 | 0.011282 | 0.4286 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma v4 | launch | 0.006130 | 0.018506 | 0.3312 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma v4 | launch | 0.006104 | 0.018790 | 0.3248 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma v4 | launch | 0.012864 | 0.045853 | 0.2805 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma v4 | launch | 0.010270 | 0.044784 | 0.2293 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma v4 | launch | 0.032122 | 0.288206 | 0.1115 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma v4 | launch | 0.032138 | 0.285638 | 0.1125 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma v4 | launch | 0.086026 | 0.973683 | 0.0884 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma v4 | launch | 0.089285 | 0.969744 | 0.0921 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma v4 | launch | 0.195018 | 2.217618 | 0.0879 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma v4 | launch | 0.206362 | 2.218437 | 0.0930 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma v4 | launch | 0.009414 | 0.035232 | 0.2672 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma v4 | launch | 0.009493 | 0.036061 | 0.2632 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma v4 | launch | 0.008248 | 0.035470 | 0.2325 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma v4 | launch | 0.008613 | 0.035944 | 0.2396 | 0.00075151 | 0 |

- **Analysis:** Quick signal only, not final acceptance; protocol differs from iter7full100/5/3. Matched Ltbatch approximate geometric latency gain=0.4538x. 128 bf16=6.86us (Dense/SW 0.607x); 128 fp16=7.09us (Dense/SW 0.582x); 256 bf16=11.16us (Dense/SW 0.471x); 256 fp16=11.28us (Dense/SW 0.429x); 512 bf16=18.51us (Dense/SW 0.331x); 512 fp16=18.79us (Dense/SW 0.325x); 1024 bf16=45.85us (Dense/SW 0.281x); 1024 fp16=44.78us (Dense/SW 0.229x). Tiny128 improves absolute latency relative to batch, while larger tiles regress as repeated DAG conversion and smallK steps dominate. All18freshchecks fail frozen numeric gate with finite outputs; AutoDense.
- **Next:** GPU structural/sanitizer and Graph tests; profile SIMT cast work, compare warp grouping, then paired finalcasts
- **Archive:** trajectory/20261013_132604_iter-9/

### Iter 10 — Specialize complete small tiles and packed final conversions

- **Hypothesis:** Remove generic vector-bound checks and 64-bit owner division/modulo while preserving complete products and fixed FP32 U DAG.
- **Changes:** Add independent Mega 7 packed RN casts and Mega 8 complete-tile 2D grid path; zero workspace; preserve Mega 4-6; add exact and bitwise tests; add final small suite and verification scripts.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.399807 ms; mean speedup=0.3431x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.004208 | 0.005277 | 0.7975 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.004123 | 0.005403 | 0.7631 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.005496 | 0.008198 | 0.6704 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.004866 | 0.008190 | 0.5941 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.006165 | 0.013720 | 0.4493 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.006166 | 0.013912 | 0.4432 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.012861 | 0.041330 | 0.3112 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.010254 | 0.041168 | 0.2491 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.031942 | 0.284176 | 0.1124 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.032947 | 0.284205 | 0.1159 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.086251 | 0.974886 | 0.0885 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.089274 | 0.974491 | 0.0916 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.194622 | 2.205798 | 0.0882 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.205419 | 2.225819 | 0.0923 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.009474 | 0.027360 | 0.3463 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.009578 | 0.027531 | 0.3479 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_fast v8 | launch | 0.008266 | 0.027304 | 0.3027 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_fast v8 | launch | 0.008662 | 0.027750 | 0.3122 | 0.00075151 | 0 |

- **Analysis:** Quick predeclared 20 calls x 5 groups x 1 fresh correctness trial, H100, ordinary launch, variant 8. Matched geometric gain vs iteration 9 variant 4 = 1.181938x; bf16 128^3 5.277 us Dense/SW=0.797453010639; fp16 128^3 5.403 us Dense/SW=0.763103342123; bf16 256^3 8.198 us Dense/SW=0.670374735566; fp16 256^3 8.190 us Dense/SW=0.594061349577; bf16 512^3 13.720 us Dense/SW=0.449329453166; fp16 512^3 13.912 us Dense/SW=0.443243251928; bf16 1024^3 41.330 us Dense/SW=0.311176490651; fp16 1024^3 41.168 us Dense/SW=0.249086666856. All 18 numerical gates fail with finite output (exit 2); frozen thresholds unchanged. Paired conversion is an ablation because iteration 9 SASS already contains packed conversion instructions. Structural GPU/sanitizer validation follows this closure.
- **Next:** Validate variants 7/8 and compare packed ablation; integrate product-parallel M1 variant 9 with seven complete products.
- **Archive:** trajectory/20261013_133924_iter-10/

### Iter 11 — Parallelize seven complete products across warps and fix epilogue rounding

- **Hypothesis:** Seven product warps reduce56live FP32 accumulators per lane to8 and expose latency hiding while retaining complete-P then U semantics.
- **Changes:** Add Mega9 product-parallel M1; 224threads,14KiBshared,0workspace; explicit alpha RN followed by beta FMA for Mega4-8; strengthen common exact/bitwise route tests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=1.033407 ms; mean speedup=0.2446x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_product_warps v9 | launch | 0.004181 | 0.005942 | 0.7036 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_product_warps v9 | launch | 0.004170 | 0.006107 | 0.6827 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_product_warps v9 | launch | 0.005322 | 0.008578 | 0.6204 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_product_warps v9 | launch | 0.004883 | 0.008669 | 0.5633 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_product_warps v9 | launch | 0.006160 | 0.018938 | 0.3253 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_product_warps v9 | launch | 0.006144 | 0.018922 | 0.3247 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_product_warps v9 | launch | 0.013122 | 0.097243 | 0.1349 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_product_warps v9 | launch | 0.010192 | 0.096216 | 0.1059 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_product_warps v9 | launch | 0.032205 | 0.721770 | 0.0446 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_product_warps v9 | launch | 0.032342 | 0.732958 | 0.0441 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_product_warps v9 | launch | 0.086350 | 2.487458 | 0.0347 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_product_warps v9 | launch | 0.089195 | 2.497160 | 0.0357 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_product_warps v9 | launch | 0.195328 | 5.819426 | 0.0336 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_product_warps v9 | launch | 0.206648 | 5.882320 | 0.0351 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_product_warps v9 | launch | 0.009173 | 0.048568 | 0.1889 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_product_warps v9 | launch | 0.009506 | 0.050253 | 0.1892 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_product_warps v9 | launch | 0.008269 | 0.050178 | 0.1648 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_product_warps v9 | launch | 0.008629 | 0.050626 | 0.1704 | 0.00075151 | 0 |

- **Analysis:** Predeclared quick20/5/1 launch signal. Primary128..1024 geometric gain vs best single-warp fast8 iter10=0.715546x. bf16 128^3 5.942us Dense/SW=0.704x; fp16 128^3 6.107us Dense/SW=0.683x; bf16 256^3 8.578us Dense/SW=0.620x; fp16 256^3 8.669us Dense/SW=0.563x; bf16 512^3 18.938us Dense/SW=0.325x; fp16 512^3 18.922us Dense/SW=0.325x; bf16 1024^3 97.243us Dense/SW=0.135x; fp16 1024^3 96.216us Dense/SW=0.106x. Product warps have66registers/thread,14KiBshared,0spill; reduced live accumulators do not offset repeated selective loads, warp branching and CTA combination. All18numeric gatesfail,0nonfinite. Prior iter10 GPU smoke7/8 and six sanitizerchecks PASS, but fresh dyadic default bitwise test fails beta nonzero due to compiler alpha/beta FMA contraction; this iter makes old4..8 alpha __fmul_rn then beta fmaf explicit like9/combine; original bitwise test retained.
- **Next:** Validate bitwise regression and product9; benchmark complete-tile product10, then narrower16x8owners.
- **Archive:** trajectory/20261013_134640_iter-11/

### Iter 12 — Remove complete-tile checks from parallel product warps

- **Hypothesis:** Unchecked aligned vector loads and int K counter shorten selective operand construction for seven product warps.
- **Changes:** Add Mega10 with complete32multiple tiles, all4ld8 guard, exact2Dgrid and generic9fallback; immutable9source; add10toexactbitwise/tail/rebindtests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.914679 ms; mean speedup=0.3034x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_product_warps_fast v10 | launch | 0.004178 | 0.004642 | 0.9000 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_product_warps_fast v10 | launch | 0.004147 | 0.004782 | 0.8672 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_product_warps_fast v10 | launch | 0.005442 | 0.006390 | 0.8515 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_product_warps_fast v10 | launch | 0.004850 | 0.006406 | 0.7570 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_product_warps_fast v10 | launch | 0.006317 | 0.013835 | 0.4566 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_product_warps_fast v10 | launch | 0.006184 | 0.013856 | 0.4463 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_product_warps_fast v10 | launch | 0.013003 | 0.107262 | 0.1212 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_product_warps_fast v10 | launch | 0.010240 | 0.107382 | 0.0954 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_product_warps_fast v10 | launch | 0.031982 | 0.678390 | 0.0471 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_product_warps_fast v10 | launch | 0.032120 | 0.681822 | 0.0471 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_product_warps_fast v10 | launch | 0.086142 | 2.218050 | 0.0388 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_product_warps_fast v10 | launch | 0.088686 | 2.261314 | 0.0392 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_product_warps_fast v10 | launch | 0.194874 | 5.071085 | 0.0384 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_product_warps_fast v10 | launch | 0.206202 | 5.094073 | 0.0405 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_product_warps_fast v10 | launch | 0.009186 | 0.048459 | 0.1896 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_product_warps_fast v10 | launch | 0.009242 | 0.048763 | 0.1895 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_product_warps_fast v10 | launch | 0.008040 | 0.048819 | 0.1647 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_product_warps_fast v10 | launch | 0.008373 | 0.048890 | 0.1713 | 0.00075151 | 0 |

- **Analysis:** Predeclared quick20/5/1 launch. Primary128..1024 geometric gain vs best fast8=0.863764x; vs product9=1.207139x. bf16 128^3 4.642us Dense/SW=0.900x; fp16 128^3 4.782us Dense/SW=0.867x; bf16 256^3 6.390us Dense/SW=0.852x; fp16 256^3 6.406us Dense/SW=0.757x; bf16 512^3 13.835us Dense/SW=0.457x; fp16 512^3 13.856us Dense/SW=0.446x; bf16 1024^3 107.262us Dense/SW=0.121x; fp16 1024^3 107.382us Dense/SW=0.095x. FullTile productfast does not overcome repeated operand loading and product-warp branch cost; retain additive comparison. All18numerical gatesfail finite. Iter11 full GPU structural/bitwise4/7/8/9 PASS; three9sanitizers0errors0warnings. Rounding fix verified without weakening tests.
- **Next:** Validate10 and integrate native16x8owners11; if third no-improvement iteration occurs, reassess profiler/compiler and primary literature before next edit.
- **Archive:** trajectory/20261013_135104_iter-12/

### Iter 13 — Reduce fused output owners to native16x8 Tensor Core tiles

- **Hypothesis:** 28persistentFP32 values/lane and twice as many ownerCTAs improve small-grid coverage despite repeated A transforms.
- **Changes:** Add Mega11 16x8x16 Sm80 MMA,5.25KiBshared,0workspace; packed7fallback; minimalN16/N48 and existing exactdyadicbitwise tests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.681080 ms; mean speedup=0.2953x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.004150 | 0.005139 | 0.8076 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.004123 | 0.005373 | 0.7674 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.005286 | 0.008170 | 0.6471 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.004830 | 0.008296 | 0.5823 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.006138 | 0.014478 | 0.4239 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.006109 | 0.014622 | 0.4178 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.012910 | 0.065978 | 0.1957 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.010141 | 0.065354 | 0.1552 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.031982 | 0.518272 | 0.0617 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.031931 | 0.525723 | 0.0607 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.086106 | 1.565026 | 0.0550 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.089197 | 1.610131 | 0.0554 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.194962 | 3.843285 | 0.0507 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.205952 | 3.867382 | 0.0533 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.009240 | 0.035670 | 0.2590 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.009275 | 0.035597 | 0.2606 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_16x8 v11 | launch | 0.008048 | 0.035456 | 0.2270 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_16x8 v11 | launch | 0.008368 | 0.035480 | 0.2359 | 0.00075151 | 0 |

- **Analysis:** Predeclared quick20/5/1 launch signal. Primary128..1024 geometric gain vs prior observed envelope=0.642482x (prior includes full100-call batch samples, signal-only cross-protocol comparison; final same-binary full repeat required). bf16 128^3 5.139us Dense/SW=0.808x best-prior/SW=0.903x; fp16 128^3 5.373us Dense/SW=0.767x best-prior/SW=0.890x; bf16 256^3 8.170us Dense/SW=0.647x best-prior/SW=0.782x; fp16 256^3 8.296us Dense/SW=0.582x best-prior/SW=0.772x; bf16 512^3 14.478us Dense/SW=0.424x best-prior/SW=0.820x; fp16 512^3 14.622us Dense/SW=0.418x best-prior/SW=0.807x; bf16 1024^3 65.978us Dense/SW=0.196x best-prior/SW=0.301x; fp16 1024^3 65.354us Dense/SW=0.155x best-prior/SW=0.300x. Narrow owner doubles CTA coverage and halves P fragment, but repeats A transforms and retains B lane<16 instruction sequence. All18numerical gatesfail finite. Iter12 complete GPU structural/bitwise4/7/8/9/10 PASS and three10sanitizers0errors0warnings. Reassess current compiler/timeline, primary CUDA/CUTLASS docs and logs before next source iteration; viable next B all-lane four-element vector optimization is already CPU-reviewed.
- **Next:** Reassess and GPUvalidate11; then measure all-lane four-element B fill variant12.
- **Archive:** trajectory/20261013_135613_iter-13/

### Reassessment after iterations 11–13 (small matrices)

The aggregate128–1024 score of each new path has not exceeded the best prior paths for three trials; variant10 nevertheless improved128/256 and remains the tiny winner. Thus no blanket plateau or hardware floor is inferred from this aggregate. The logs preserve the per-shape distinction. The current H100 is clock-unlocked; comparing own latency is an iteration signal, while same-binary100/5/3 alternating Dense/candidate and separate launch/Graph are required at closure.

Nsight Systems128 BF16 traces show one GPU kernel and no memory-transfer activity: productfast10=4.032us; native16x8_11=4.512us. The profiled host API duration is heavily inflated and is not used as steady-state overhead. These kernel durations are close to, but below, their ordinary-launch event measurements4.642/5.139us: GPU work remains material, so a launch-floor stop is unsupported. ncu remains unavailable after the original single failed probe; it is not retried. Full GPU structural/dyadic bitwise4/7/8/9/10/11 passes; variant11 memcheck/racecheck/synccheck have0errors0warnings.

Re-read primary sources: [CUDA12.4.1 instruction/control-flow guidance](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#maximize-instruction-throughput), [CUTLASS3.5.1 native Sm80 MMA definitions](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch/mma_sm80.hpp), [CUDA12.4.1 DSM lifecycle](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#distributed-shared-memory), and [explicit rounding/FMA control](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/Floating_Point_on_NVIDIA_GPU.pdf). These support instruction reduction and synchronization design, not a promised speedup. Lower-precision half2 preadds would violate the FP32 S/T contract and are excluded.

Viable directions remain and are CPU-reviewed: (1) all32 B lanes each4elements instead of half-warp8elements to halve issued conversion work; (2) cluster8 with one product per CTA to spread sparse tiny work while protecting DSM with two cluster barriers; (3) float2 output and32bit tile counters,64bit stride products; (4) immutable alpha1/beta0 output specialization. Continue testing these independently before final full repeats. Evidence: results/timeline_small_mma10_reassess.txt, results/timeline_small_mma11_reassess.txt, results/small_sass_reassessment_iter13.json and trajectory iterations11–13.

### Iter 14 — 16x8 B fill across a full warp

- **Hypothesis:** Assign four B elements to every lane instead of eight to half the warp to reduce issued conversion and predicate work while preserving the fixed SW DAG.
- **Changes:** Added Mega12 with uint2 B loads and shared stores, two final RN conversion pairs per lane, the existing full-tile guards, zero workspace, and explicit alpha RN then beta FMA; retained Mega11 and all earlier variants.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.678265 ms; mean speedup=0.3673x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.005992 | 0.004992 | 1.2003 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.005981 | 0.005179 | 1.1548 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.006013 | 0.007541 | 0.7974 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.005990 | 0.007566 | 0.7917 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.006549 | 0.013683 | 0.4786 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.006619 | 0.013669 | 0.4843 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.013790 | 0.066320 | 0.2079 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.010413 | 0.066200 | 0.1573 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.032184 | 0.522301 | 0.0616 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.033318 | 0.514770 | 0.0647 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.086234 | 1.610298 | 0.0536 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.089186 | 1.575963 | 0.0566 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.195517 | 3.830817 | 0.0510 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.206056 | 3.829974 | 0.0538 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.009198 | 0.034874 | 0.2638 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.009229 | 0.034979 | 0.2638 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.008027 | 0.034798 | 0.2307 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_16x8_vec4 v12 | launch | 0.008334 | 0.034853 | 0.2391 | 0.00075151 | 0 |

- **Analysis:** Quick 20-call five-group screening completed for all 18 rows; all rows are finite and all frozen numerical gates fail, yielding exit 2. Results are archived at trajectory/20261013_140256_iter-14. Performance must be compared per primary shape against the prior best envelope after closure; this quick run is a screening signal and does not certify speedup. The preceding CUDA/CUTLASS, SASS, and Nsight Systems reassessment identified remaining measurable GPU work and further viable directions.
- **Next:** Validate Mega12 structural and bitwise behavior with all three sanitizers, then test portable eight-CTA product clusters as the next separate iteration.
- **Archive:** trajectory/20261013_140256_iter-14/

Iter14 post-closure primary analysis (quick screening, mixed historical protocols, no certification): bf16 128^3 4.992us vs priorbest iter12 4.642us; prior/SW=0.930x; same-row Dense/SW=1.200x; fp16 128^3 5.179us vs priorbest iter12 4.782us; prior/SW=0.923x; same-row Dense/SW=1.155x; bf16 256^3 7.541us vs priorbest iter12 6.390us; prior/SW=0.847x; same-row Dense/SW=0.797x; fp16 256^3 7.566us vs priorbest iter12 6.406us; prior/SW=0.847x; same-row Dense/SW=0.792x; bf16 512^3 13.683us vs priorbest iter8 11.872us; prior/SW=0.868x; same-row Dense/SW=0.479x; fp16 512^3 13.669us vs priorbest iter8 11.805us; prior/SW=0.864x; same-row Dense/SW=0.484x; bf16 1024^3 66.320us vs priorbest iter7 19.830us; prior/SW=0.299x; same-row Dense/SW=0.208x; fp16 1024^3 66.200us vs priorbest iter7 19.638us; prior/SW=0.297x; same-row Dense/SW=0.157x. Complete GPU structural/bitwise4,7,8,9,10,11,12 PASS; variant12 memcheck/racecheck/synccheck each zero errors and warnings. Same-row Dense128 is ~6us here versus ~4us in previous screens, reinforcing need for final same-binary full alternating repeats.

Iter14 compiler mechanism check (same3902367 binary): native16x8 vec4 has24staticpacked final casts vs32 for variant11, 655static instructions vs679; both7HMMA/K16. This confirms narrower B conversion issue count but does not imply dynamic utilization or a speedup. Resource blocks and binary hash recorded in results/small_sass_iter14.json.

### Iter 15 — Spread seven complete products over eight-CTA Hopper clusters

- **Hypothesis:** An eight-CTA portable cluster can spread sparse tiny product work over more SMs and combine completed FP32 P values through DSM.
- **Changes:** Added Mega13 product cluster with seven single-warp product CTAs, one barrier-only CTA, two unconditional cluster barriers, rank0 fixed U, zero workspace, prepare-only capability queries, and generic Mega9 fallback. Added explicit Graph replay tests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=1.616020 ms; mean speedup=0.2237x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_product_cluster v13 | launch | 0.005678 | 0.006624 | 0.8572 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_product_cluster v13 | launch | 0.005610 | 0.006838 | 0.8203 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_product_cluster v13 | launch | 0.005656 | 0.009632 | 0.5872 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_product_cluster v13 | launch | 0.005709 | 0.009618 | 0.5936 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_product_cluster v13 | launch | 0.006387 | 0.030488 | 0.2095 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_product_cluster v13 | launch | 0.006402 | 0.030726 | 0.2083 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_product_cluster v13 | launch | 0.013498 | 0.152165 | 0.0887 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_product_cluster v13 | launch | 0.010370 | 0.151435 | 0.0685 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_product_cluster v13 | launch | 0.032067 | 1.005120 | 0.0319 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_product_cluster v13 | launch | 0.032526 | 1.022578 | 0.0318 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_product_cluster v13 | launch | 0.086064 | 3.204152 | 0.0269 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_product_cluster v13 | launch | 0.088834 | 3.239051 | 0.0274 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_product_cluster v13 | launch | 0.196355 | 9.857846 | 0.0199 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_product_cluster v13 | launch | 0.206280 | 10.042958 | 0.0205 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_product_cluster v13 | launch | 0.009154 | 0.079622 | 0.1150 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_product_cluster v13 | launch | 0.009206 | 0.080040 | 0.1150 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_product_cluster v13 | launch | 0.008006 | 0.079578 | 0.1006 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_product_cluster v13 | launch | 0.008347 | 0.079896 | 0.1045 | 0.00075151 | 0 |

- **Analysis:** Build and quick 20-call five-group screening completed for all 18 finite rows; all frozen numerical gates failed, exit2. Per-shape performance must be compared against the prior best envelope after closure, and final acceptance requires the full same-binary protocol. No numerical threshold or product completion order changed.
- **Next:** Run complete structural/bitwise and Graph tests plus cluster memcheck/racecheck/synccheck, then test the independent float2 output epilogue.
- **Archive:** trajectory/20261013_141750_iter-15/
