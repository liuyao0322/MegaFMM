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

| 16 | Vectorize native16x8 output and use bounded int tile counters | 0.4140 | 0.678297 | screened; numerical gate failed; output validation pending |

| 17 | Specialize native16x8 output for immutable alpha1 beta0 | 0.3082 | 0.678775 | screened; numerical gate failed; identity validation pending |

| 18 | Prefetch next raw K16 tile into registers | 0.3553 | 0.675800 | promising-screen-numerical-fail |

| 19 | Compact seven-batch BK64 pipeline controls | 0.4534 | 0.122446 | screen-numerical-fail |

| 20 | Double-buffer raw quadrants through cp.async | 0.3526 | 0.497472 | screen-numerical-fail |

| 21 | Strengthen smallLt tuning and record actual bound algorithms | 0.5287 | 0.072138 | measurement-traceability-numerical-fail |

| 22 | Accumulate per-stage Winograd contributions in four C fragments | 0.3137 | 0.426841 | screen-numerical-fail |

| 23 | Split prefetched warp owners into eight-by-eight outputs | 0.3041 | 1.037819 | screen-numerical-fail |

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

Iter15 post-closure primary comparison (quick signal, some historicalfull protocols, no acceptance): bf16 128^3 6.624us Dense/SW=0.857x prior(iter12)/SW=0.701x; fp16 128^3 6.838us Dense/SW=0.820x prior(iter12)/SW=0.699x; bf16 256^3 9.632us Dense/SW=0.587x prior(iter12)/SW=0.663x; fp16 256^3 9.618us Dense/SW=0.594x prior(iter12)/SW=0.666x; bf16 512^3 30.488us Dense/SW=0.209x prior(iter8)/SW=0.389x; fp16 512^3 30.726us Dense/SW=0.208x prior(iter8)/SW=0.384x; bf16 1024^3 152.165us Dense/SW=0.089x prior(iter7)/SW=0.130x; fp16 1024^3 151.435us Dense/SW=0.068x prior(iter7)/SW=0.130x.

Iter15 GPU closure: complete structural/bitwise4,7,8,9,10,11,12,13 and independent clusterGraphreplay PASS; CUDA12.4memcheck andsynccheck PASS. CUDA12.4racecheck abortedwithhostSIGSEGV(exit11), genericUnifiedMemorydiagnostic beforefirstcasefinish; zero reportedhazards is not a pass. Same unchanged CUDA12.4binary passed cluster racecheck under standalone CUDA12.8ComputeSanitizer2025.1.0.0, zeroerrors/warnings/hazards includingGraphreplays. Preserve failedlog andsuccessfulnewertoollog. Finalverification now supports explicitMEGAFMM_SANITIZER and records toolpath/version, whilebuild/runtime stayCUDA12.4.

Iter15 once-only NsightSystems128BF13: GPUkernel5.792us vs6.624us ordinaryquickmedian; one kernel andnoGPUmemoryoperations. Captured API~91us is instrumentation-inflated, notsteadyhostlatency. This gives no measuredbreakdown ofDSM/barriers andno physicalfloor; retain as direction-cost evidence. Source28e9b15/binaryhash recordedinresults/small_cluster_profile_iter15.json.

### Iter 16 — Vectorize native16x8 output and use bounded int tile counters

- **Hypothesis:** Adjacent native16x8 output pairs permit float2 C0 loads and D stores, with shorter tile address arithmetic and unchanged arithmetic order.
- **Changes:** Added Mega14 float2 epilogue from Mega12, explicit output and active-C0 alignment/stride guards, int coordinates with int64 stride products, and per-run fallback14 to12 to7. Added output/C0 four-byte rebinding, odd output/source strides, beta0 unused-C0, alias and bitwise tests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.678297 ms; mean speedup=0.4140x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007619 | 0.005347 | 1.4249 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007360 | 0.005530 | 1.3310 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007648 | 0.008098 | 0.9445 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007624 | 0.008126 | 0.9382 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007640 | 0.014293 | 0.5345 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.007706 | 0.014326 | 0.5379 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.014440 | 0.065474 | 0.2205 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.010480 | 0.065350 | 0.1604 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.032250 | 0.517125 | 0.0624 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.032979 | 0.523274 | 0.0630 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.087059 | 1.595080 | 0.0546 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.089547 | 1.572221 | 0.0570 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.194594 | 3.821192 | 0.0509 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.205878 | 3.852611 | 0.0534 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.009472 | 0.035200 | 0.2691 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.009562 | 0.035357 | 0.2704 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.008283 | 0.035434 | 0.2338 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_16x8_epilogue v14 | launch | 0.008638 | 0.035307 | 0.2447 | 0.00075151 | 0 |

- **Analysis:** Quick20/5/1 screening completed for18 finite rows; all frozen numerical gates failed, exit2. This is diagnostic performance evidence and not certified speedup. All older variants remain available. Cluster13 full structural/bitwise/Graph/mem/sync passed; racecheck host crash onCUDA12.4 was preserved and resolved with standaloneCUDA12.8 on the sameCUDA12.4binary.
- **Next:** Validate14 including pointer rebinding and three sanitizers, inspect generated vector stores and resources, and reassess logs/compiler/timeline/primary sources before the next identity specialization.
- **Archive:** trajectory/20261013_142612_iter-16/

Iter16 post-closure primary comparison (quick signal, some historical full protocols, no acceptance): bf16 128^3 5.347us Dense/SW=1.425x prior(iter12)/SW=0.868x; fp16 128^3 5.530us Dense/SW=1.331x prior(iter12)/SW=0.865x; bf16 256^3 8.098us Dense/SW=0.944x prior(iter12)/SW=0.789x; fp16 256^3 8.126us Dense/SW=0.938x prior(iter12)/SW=0.788x; bf16 512^3 14.293us Dense/SW=0.535x prior(iter8)/SW=0.831x; fp16 512^3 14.326us Dense/SW=0.538x prior(iter8)/SW=0.824x; bf16 1024^3 65.474us Dense/SW=0.221x prior(iter7)/SW=0.303x; fp16 1024^3 65.350us Dense/SW=0.160x prior(iter7)/SW=0.300x.

### Reassessment after iterations14–16 (before17)

No primary128/256/512/1024 shape improved the prior observed SW envelope by3% in three consecutive sourceiterations:14 all-lane B fill,15 productcluster,16 float2/int-coordinate output. Prior envelope still10 on128/256 andH5/Lt1 on512/1024; mixed historical20/100-call runs remain signals requiring finalsamebinary repeats. Re-read these per-shape logs and once-only timelines:13clusterGPU5.792us,14epilogueGPU4.352us, versus earlier10GPU4.032us at128BF. InstrumentedhostAPIduration is notsteady launchlatency. No timer/bandwidth/launchphysicalfloor is established. Complete14 structural/bitwise/rebind and three CUDA12.8sanitizers passed zeroerrors/warnings.

Compiled14 preserves24packedcasts and7HMMA/K16, now16STG.E.64 instead of12's32scalarstores;583staticinstructions vs655,94registers, noLOCAL/STACK. Static instruction reduction alone did not improve the bestlatency. Source405376f, currentbinaryhash/resourceblocks and successful epilogue profile capture inresults/small_reassessment_iter16.json. Dense128quickmedian jumped again (~7.6us), so reference ratios alone are insufficient to rank kernel changes on unlockedclocks; ownlatency and finalGraph/controlmeasurements remainessential. Two profiler attempts using --algorithm dense and --algorithm auto were rejected beforecapture because the CLI accepts only lt/hopper/mega. Both failedlogs are preserved; no Dense GPU duration was measured. Future final profiling needs an explicit Dense capture option, and this tooling limitation is not a stop reason.

Primary-source reassessment: CUDA12.4.1 BestPractices async-copy avoids intermediate global-to-shared registers and permits overlap; implementation must wait beforeconsuming and preserve currentbank lifetime (https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory). Pinned CUTLASS3.5.1 MmaPipelined is two-stage and excludes two warpK groups, hence proposedwarpK64 has four instructionK16groups (https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/mma_pipelined.h).

Viable separately reviewed directions remain: immutablealpha1/beta0 specialization(17/Mega15); rawregister lookahead(18/Mega16) with SASSissue/cache-policy checks; compactseven-batch BK64stage3/stage2 controls(19/H7,H8); rawquadrant cp.async double-bank16x16 owner(20/Mega17). Continue these rather than claimexhaustion or a provenfloor. Strengthen smallLt setup-onlytuning and recordactualboundconfiguration in a separatefinalsourceiteration beforethefullcomparison; do not change32heuristics/budget/dedup/gates.

### Iter 17 — Specialize native16x8 output for immutable alpha1 beta0

- **Hypothesis:** A plan with alpha exactly one and beta zero can write U directly and omit coefficient operations and C0 addresses without changing its bit patterns.
- **Changes:** Added Mega15 identity-only native16x8 path and guarded chain15 to14 to12 to7 using actual rebound pointers. Retained general coefficients and all earlier routes; extended identity, signed-zero, alias, stride, rebind and bitwise tests.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.678775 ms; mean speedup=0.3082x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.004245 | 0.004925 | 0.8619 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.004187 | 0.005109 | 0.8196 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.005285 | 0.007842 | 0.6739 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.004838 | 0.007976 | 0.6066 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.006131 | 0.014005 | 0.4378 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.006150 | 0.014109 | 0.4359 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.012888 | 0.064856 | 0.1987 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.010227 | 0.064760 | 0.1579 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.032283 | 0.521126 | 0.0619 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.032965 | 0.511320 | 0.0645 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.086144 | 1.603035 | 0.0537 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.089424 | 1.598578 | 0.0559 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.195013 | 3.831363 | 0.0509 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.206277 | 3.831589 | 0.0538 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.009206 | 0.034064 | 0.2703 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.009256 | 0.034430 | 0.2688 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.008024 | 0.034414 | 0.2332 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_16x8_identity v15 | launch | 0.008358 | 0.034454 | 0.2426 | 0.00075151 | 0 |

- **Analysis:** Build and quick20/5/1 screening completed with18 finite rows; all frozen numerical gates fail, exit2. Dynamic input transforms remain inside every call. This screening is not full acceptance. The earlier 14-16 stall reassessment is logged, with GPU timeline and static SASS evidence and remaining separately reviewed directions.
- **Next:** Run full structural/bitwise and three15sanitizers, then test raw register lookahead as a separate iteration.
- **Archive:** trajectory/20261013_144105_iter-17/

Iter17 post-closure primary comparison (quick signal, mixed historical full protocols, no acceptance): bf16 128^3 4.925us Dense/SW=0.862x prior(iter12)/SW=0.942x; fp16 128^3 5.109us Dense/SW=0.820x prior(iter12)/SW=0.936x; bf16 256^3 7.842us Dense/SW=0.674x prior(iter12)/SW=0.815x; fp16 256^3 7.976us Dense/SW=0.607x prior(iter12)/SW=0.803x; bf16 512^3 14.005us Dense/SW=0.438x prior(iter8)/SW=0.848x; fp16 512^3 14.109us Dense/SW=0.436x prior(iter8)/SW=0.837x; bf16 1024^3 64.856us Dense/SW=0.199x prior(iter7)/SW=0.306x; fp16 1024^3 64.760us Dense/SW=0.158x prior(iter7)/SW=0.303x.

### Iter 18 — Prefetch next raw K16 tile into registers

- **Hypothesis:** Issuing next-tile global loads before current seven products can overlap memory latency without changing any transformed operand or P/U accumulation order.
- **Changes:** Added Mega16 raw 24-u32 bank lookahead, guarded identity chain16 to15 to14 to12 to7, zero workspace, exact current input pointers, and short K32/64/96, N16/N48, bitwise, coefficient and rebinding tests. Earlier variants retained verbatim.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.675800 ms; mean speedup=0.3553x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.004203 | 0.004126 | 1.0186 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.004126 | 0.004339 | 0.9510 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.005286 | 0.006346 | 0.8331 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.004827 | 0.006504 | 0.7422 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.006179 | 0.010994 | 0.5621 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.006141 | 0.011086 | 0.5539 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.012928 | 0.065475 | 0.1974 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.010210 | 0.065357 | 0.1562 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.031547 | 0.486712 | 0.0648 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.032810 | 0.486224 | 0.0675 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.085768 | 1.624280 | 0.0528 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.089288 | 1.620158 | 0.0551 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.194886 | 3.801155 | 0.0513 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.205702 | 3.836810 | 0.0536 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.009197 | 0.033747 | 0.2725 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.009262 | 0.033685 | 0.2750 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.008066 | 0.033794 | 0.2387 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_16x8_prefetch v16 | launch | 0.008384 | 0.033611 | 0.2494 | 0.00075151 | 0 |

- **Analysis:** Quick20/5/1 screening completed18 finite rows, all frozen numerical gates fail, exit2. Initial128BF grouping is4.13–4.26us versus historical4.64us best, a promising screening signal requiring same-binary full confirmation; compiler uses120registers without spills. This variant can also change default inlinePTX global cache policy relative to readonly compiler loads, so actual SASS issue order and cache modifiers must be inspected before attributing gains purely to scheduling.
- **Next:** Validate16 structural/bitwise/short-K with three standaloneCUDA12.8sanitizers and SASS, then separately test compact BK64 seven-batch H7/H8.
- **Archive:** trajectory/20261013_144926_iter-18/

Iter18 post-closure primary comparison (quick20/5/1, mixed historical protocols, not acceptance): bf16 128^3 4.126us Dense/SW=1.019x priorbest/SW=1.125x; fp16 128^3 4.339us Dense/SW=0.951x priorbest/SW=1.102x; bf16 256^3 6.346us Dense/SW=0.833x priorbest/SW=1.007x; fp16 256^3 6.504us Dense/SW=0.742x priorbest/SW=0.985x; bf16 512^3 10.994us Dense/SW=0.562x priorbest/SW=1.080x; fp16 512^3 11.086us Dense/SW=0.554x priorbest/SW=1.065x; bf16 1024^3 65.475us Dense/SW=0.197x priorbest/SW=0.303x; fp16 1024^3 65.357us Dense/SW=0.156x priorbest/SW=0.300x. 128 both dtypes and512 both dtypes improve the prior observed latency by more than3%;256 within noise/no3% gain,1024 remains slower thanLt1. Full same-binary launch/Graph repeats and actual loading/cache-policy inspection required.

### Iter17 post-closure identity15 validation and static SASS evidence

CPU-only audit of archived iter16/17 SASS/resources and iter17 logs is saved in [results/small_identity_profile_iter17.json](results/small_identity_profile_iter17.json). The requested small_sass_iter16.json was absent; the comparison uses the existing raw iter16 SASS/resources and records small_reassessment_iter16.json metadata. Mega14 parsed instructions/resources are unchanged across16→17 for BF16 and FP16. Mega15 has96 registers versus14's94; both report6400 bytes compiled static shared, zero STACK/LOCAL, and no LDL/STL. Source declares5376 bytes of operand shared; the additional1024 compiled bytes are recorded without assigning an unverified cause.

The complete instruction parser includes one always-false @!UPT UIADD3 omitted by the older parser, giving584 static instructions for14 and440 for15 (older counts583/439); this is a parser difference, not a code change. Identity15 has zero FMUL/FFMA/beta comparison/C0 load path. It preserves four LDG.E.128.CONSTANT A loads, four LDG.E.64.CONSTANT B loads, seven HMMA, seven A LDSM.x4, seven B transposed LDSM.x2 and24 packed RN converters per K16; cache modifiers remain the same. It writes eight STG.E.64 float2 vectors per lane. Mega14's static16 stores include two alternative beta paths, each executing eight; beta0 already skipped its C0 loads. Thus the specialization removes sixteen executed alpha multiplies and beta branch/address code, without evidence of halving matched beta0 memory traffic. The static144-instruction reduction does not establish dynamic speedup, occupancy or a physical floor.

correctness_iter17.txt has936 finite CASE rows (468/dtype),85 explicit PASS markers,518 expected/actual route checks with zero mismatches, and58 Mega15 CASE rows across identity/epilogue/vec4/packed/Dense fallbacks with all four reported error metrics and nonfinite equal zero. Bitwise tests cover both dtypes, general coefficients/alias and identity; signed-zero and input/output rebinding are covered. Each identity15 sanitizer has26 exact structural CASE rows plus ten output-rebind PASS markers: memcheck0 errors, racecheck0 errors/0 warnings, synccheck0 errors. The audit observed complete PASS/summary footers, not independent exit-status files. These tests do not certify the frozen mixed-precision gate or identity15-specific Graph replay; quick17's18 gate failures remain unchanged. No GPU/build/benchmark/git action or source/script edit was performed for this evidence audit.

### Iter 19 — Compact seven-batch BK64 pipeline controls

- **Hypothesis:** A narrow32x64 BK64 leaf CTA can increase small-problem coverage while pipelining four instructionK16groups.
- **Changes:** Added Hopper7 three-stage MmaMultistage cp.async and Hopper8 two-stage MmaPipelined register-double-buffer controls, compact seven-batch layout, zero leaf scratch, unchanged parent pack/product/combine, partialK8/24/72 and alignment/alias/tune tests; retained5/6.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.122446 ms; mean speedup=0.4534x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.004203 | 0.009912 | 0.4241 | 0.00605216 | 0 |
| 128,128,128 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.004128 | 0.010360 | 0.3985 | 0.000744325 | 0 |
| 256,256,256 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.005293 | 0.010939 | 0.4838 | 0.00597914 | 0 |
| 256,256,256 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.004806 | 0.010954 | 0.4388 | 0.000737165 | 0 |
| 512,512,512 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.006221 | 0.011752 | 0.5293 | 0.00598238 | 0 |
| 512,512,512 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.006126 | 0.011680 | 0.5245 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.012917 | 0.022458 | 0.5752 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.010221 | 0.022424 | 0.4558 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.032230 | 0.098659 | 0.3267 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.032507 | 0.099299 | 0.3274 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.085637 | 0.281874 | 0.3038 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.090166 | 0.284144 | 0.3173 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.204154 | 0.617203 | 0.3308 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.207790 | 0.645467 | 0.3219 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.011443 | 0.017472 | 0.6549 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.010376 | 0.016552 | 0.6269 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | winograd_sm80_batched_k64 v7 | launch | 0.009013 | 0.016437 | 0.5483 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | winograd_sm80_batched_k64 v7 | launch | 0.009440 | 0.016443 | 0.5741 | 0.00075151 | 0 |

- **Analysis:** H7 quick20/5/1 completed18 finite rows, all frozen numerical gates fail, exit2. H7/H8 compare both pipeline implementation and global-load path, not a pure stage-count ablation. Parent owns A/B/P workspace; zero workspace applies only to leaf helpers. Per-shape prior-envelope comparison and H8 same-binary screening follow closure.
- **Next:** Screen H8 using unchanged binary, validate both routes with full structural tests and three standaloneCUDA12.8sanitizers, then test raw cp.async two-bank Mega17.
- **Archive:** trajectory/20261013_145706_iter-19/

Iter19 post-closure H7/H8 quick20/5/1 signal on same binary: H7: bf16 128^3 9.912us Dense/SW=0.424x priorbest/SW=0.416x; fp16 128^3 10.360us Dense/SW=0.398x priorbest/SW=0.419x; bf16 256^3 10.939us Dense/SW=0.484x priorbest/SW=0.580x; fp16 256^3 10.954us Dense/SW=0.439x priorbest/SW=0.585x; bf16 512^3 11.752us Dense/SW=0.529x priorbest/SW=0.935x; fp16 512^3 11.680us Dense/SW=0.525x priorbest/SW=0.949x; bf16 1024^3 22.458us Dense/SW=0.575x priorbest/SW=0.883x; fp16 1024^3 22.424us Dense/SW=0.456x priorbest/SW=0.876x | H8: bf16 128^3 10.243us Dense/SW=0.555x priorbest/SW=0.403x; fp16 128^3 10.698us Dense/SW=0.514x priorbest/SW=0.406x; bf16 256^3 11.389us Dense/SW=0.513x priorbest/SW=0.557x; fp16 256^3 11.438us Dense/SW=0.495x priorbest/SW=0.560x; bf16 512^3 12.890us Dense/SW=0.505x priorbest/SW=0.853x; fp16 512^3 12.822us Dense/SW=0.498x priorbest/SW=0.865x; bf16 1024^3 29.154us Dense/SW=0.461x priorbest/SW=0.680x; fp16 1024^3 29.254us Dense/SW=0.356x priorbest/SW=0.671x. Neither path improves a primary prior SW envelope by3%. All36rows finite and frozen numeric gatesfail; H8 direct screening uses the unchanged binary built from source36d9b45; concurrent next-iteration source edits do not modify that executable. No separate source iteration was introduced by this control run. Full finalprotocol still required.

Iter19 validation: full structural/API/M1bits through16 PASS; Hopper7 and8 each14isolatedcases including leafK8/24/72 and paddedalpha/betaalias, plus samePlaninputpointerrebind. StandaloneCUDA12.8 memcheck/synccheck zeroerrors; racecheck zerohazards/errors/warnings for both. Allsixsuccessful logs retained.

### Iter 20 — Double-buffer raw quadrants through cp.async

- **Hypothesis:** Global-to-shared async copies into two raw banks can overlap the next tile with current transforms and seven complete products while avoiding raw register residency.
- **Changes:** Added Mega17 one-warp16x16 owner with two4KiBrawbanks and7KiBtransformedoperands, ca-policy cp.async commit/wait/syncwarp, general alpha/beta scalar output, actualinputguard fallback7, shortK/bankflip/alias/rebind/Graph/bitwise4..17 coverage. Older paths retained.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.497472 ms; mean speedup=0.3526x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.004147 | 0.004566 | 0.9082 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.004136 | 0.004787 | 0.8640 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.005283 | 0.006354 | 0.8315 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.004813 | 0.006522 | 0.7380 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.006182 | 0.013378 | 0.4621 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.006120 | 0.013221 | 0.4629 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.012899 | 0.051638 | 0.2498 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.010122 | 0.051066 | 0.1982 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.031829 | 0.360085 | 0.0884 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.032490 | 0.360557 | 0.0901 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.086101 | 1.204648 | 0.0715 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.089114 | 1.195299 | 0.0746 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.194424 | 2.779274 | 0.0700 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.204472 | 2.780109 | 0.0735 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.009445 | 0.030638 | 0.3083 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.009486 | 0.030944 | 0.3066 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_cpasync v17 | launch | 0.008259 | 0.030813 | 0.2680 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_cpasync v17 | launch | 0.008586 | 0.030600 | 0.2806 | 0.00075151 | 0 |

- **Analysis:** Quick20/5/1 screening completed18 finite rows, all frozen numeric gatesfail exit2. This tests a16x16 owner rather than16x8 prefetch16, with extra shared raw loads and different cache policy; per-primary prior-envelope comparison follows closure. No numerical threshold or completeP/fixedU order changed.
- **Next:** Validate17 full structural/bitwise/Graph plus three standaloneCUDA12.8sanitizers and actual LDGSTS/wait/shared resources, then strengthen setup-only smallLt tuning and record bound algorithm identities before final comparison.
- **Archive:** trajectory/20261013_150459_iter-20/

### Iter18 post-closure prefetch16 source, SASS and validation evidence

Independent CPU-only analysis is frozen in [results/small_prefetch_profile_iter18.json](results/small_prefetch_profile_iter18.json). Parent reported source c029e46; current18 benchmark/test hashes were read before19 build. Identity15 remains instruction/resource-identical to17. Prefetch16 uses120 registers versus15's96, compiled static shared6400 bytes for both, zero STACK/LOCAL and no LDL/STL. Its single24-u32 raw bank is96 bytes in R62–R85, with no source shared addition; declared operand shared5376 bytes remains distinct from the6400-byte resource total.

Next-bank LDGs at0x1440..0x1660(FP16)/0x1670(BF16) are all @!P6 and precede all seven HMMA at0x1700..0x17b0. Five of fourteen current LDSM precede the first next LDG, so the compiled path overlaps some LDSM and all HMMA rather than issuing all next loads before all shared reads. No raw-dependent MOV/repack/cast occurs before the0x17c0 backedge. Every next LDG assigns encoded completion barrier5; that interval has no wait on bit5. The next-loop head0xb20 waits0x20 before raw use. Bits0..4 waits release address-read hazards before LDSM register reuse, not raw-result completion. This supports exposed lookahead without measuring whether all memory latency is hidden.

Cache modifiers change from15's LDG.E.128.CONSTANT / LDG.E.64.CONSTANT to16's LDG.E.128 / LDG.E.64. Speed cannot be attributed solely to prefetch scheduling. Seven full-K products, fixed U,24 packed RN converters and eight float2 output STG.E.64 remain. Static instructions rise440→480;16 static global-load sites consist of eight prologue plus eight guarded next-load sites, not doubled dynamic traffic. Source next_k<kh, decoded final predicate, CPU enumeration of all1024 allowed parent-K values/524800 stages, and short-K sanitizer evidence together support no final out-of-range read; static counts alone do not.

Correctness18 has1012 finite CASE rows (506/dtype),97 explicit PASS markers and574 route checks without mismatch. Its76 Mega16 CASE rows across prefetch/fallback routes all report zero errors/nonfinite. All24 bitwise records include16. Each of three16 sanitizer logs includes44 exact structural cases, with18 short-K cases spanning both dtypes,128×128/32×16/32×48 owners andK32/64/96, plus ten output-rebind PASS markers. Observed summaries are memcheck0 errors, racecheck0 errors/0 warnings, synccheck0 errors; parent separately reports all exit0. No prefetch-specific Graph PASS or strict precision certification is inferred.

The one warmed128BF NsightSystems capture observes one GPU kernel lasting3392ns (3.392us). Its instrumented host cudaLaunchKernel API lasts74266ns, which is not steady launch overhead. Quick20/5/1 medians are128 BF4.1264us/FP4.3392us and512 BF10.9936us/FP11.0864us; all18 frozen gates fail. These are promising own-latency signals requiring full same-binary confirmation. A single instrumented GPU sample, batch launch median and inflated host API duration do not form an overhead decomposition or certified speedup. No source/script edit or GPU/build/benchmark/git action was performed for this evidence audit.

Iter20 post-closure primary comparison (quick20/5/1, mixed historicalfull protocols, notacceptance): bf16 128^3 4.566us Dense/SW=0.908x priorbest/SW=0.904x; fp16 128^3 4.787us Dense/SW=0.864x priorbest/SW=0.906x; bf16 256^3 6.354us Dense/SW=0.832x priorbest/SW=0.999x; fp16 256^3 6.522us Dense/SW=0.738x priorbest/SW=0.982x; bf16 512^3 13.378us Dense/SW=0.462x priorbest/SW=0.822x; fp16 512^3 13.221us Dense/SW=0.463x priorbest/SW=0.839x; bf16 1024^3 51.638us Dense/SW=0.250x priorbest/SW=0.384x; fp16 1024^3 51.066us Dense/SW=0.198x priorbest/SW=0.385x. No primary shape beats prior observed SW envelope by3%.

Iter20 GPUvalidation: complete structural/API/M1bitwise through17 and Graph full32/full96/tail80 with3fresh replays PASS; isolation17 shortK32/64/96, generalcoeffalias/padding, input+2/output+4rebindings PASS; standaloneCUDA12.8 three17sanitizers zeromem/syncerrors and zeroracehazards/errors/warnings.

### Iter 21 — Strengthen smallLt tuning and record actual bound algorithms

- **Hypothesis:** Setup-only median-of-three ten-call candidate groups improve smallLt selection evidence while actual configuration metadata makes Dense baseline variation traceable.
- **Changes:** Small actualdescriptors≤4096 tune4warm/3x10median, largedescriptors old1warm/3calls; readonly actualbound Lt config API with9typedattrs, explicit NA/score/workspace;51columnCSV with TUNE logs outsidecapture/time and strict appendheader; exact --profile-dense single invocation; APIalias/rebindtests and manifest rawconfig/log/printing-consistency evidence.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.072138 ms; mean speedup=0.5287x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_lt_batched v1 | launch | 0.004190 | 0.009987 | 0.4196 | 0.00605216 | 0 |
| 128,128,128 | fp16 | winograd_lt_batched v1 | launch | 0.004050 | 0.010195 | 0.3972 | 0.000744325 | 0 |
| 256,256,256 | bf16 | winograd_lt_batched v1 | launch | 0.005296 | 0.011078 | 0.4780 | 0.00597914 | 0 |
| 256,256,256 | fp16 | winograd_lt_batched v1 | launch | 0.004830 | 0.010955 | 0.4409 | 0.000737165 | 0 |
| 512,512,512 | bf16 | winograd_lt_batched v1 | launch | 0.006082 | 0.012027 | 0.5057 | 0.00598238 | 0 |
| 512,512,512 | fp16 | winograd_lt_batched v1 | launch | 0.006123 | 0.011946 | 0.5126 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | winograd_lt_batched v1 | launch | 0.012822 | 0.020034 | 0.6400 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | winograd_lt_batched v1 | launch | 0.010152 | 0.019824 | 0.5121 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | winograd_lt_batched v1 | launch | 0.032293 | 0.073070 | 0.4419 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | winograd_lt_batched v1 | launch | 0.033163 | 0.067541 | 0.4910 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | winograd_lt_batched v1 | launch | 0.085387 | 0.175542 | 0.4864 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | winograd_lt_batched v1 | launch | 0.088544 | 0.154690 | 0.5724 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | winograd_lt_batched v1 | launch | 0.196011 | 0.353909 | 0.5538 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_lt_batched v1 | launch | 0.207480 | 0.305123 | 0.6800 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_lt_batched v1 | launch | 0.010440 | 0.016245 | 0.6427 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | winograd_lt_batched v1 | launch | 0.009984 | 0.015952 | 0.6259 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | winograd_lt_batched v1 | launch | 0.008464 | 0.015315 | 0.5527 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | winograd_lt_batched v1 | launch | 0.008486 | 0.015042 | 0.5642 | 0.00075151 | 0 |

- **Analysis:** Quick20/5/1 Lt1 screen completed18 finite rows, all frozen numerical gatesfailed exit2. Configurations now record actual IDs/tile/stage/index/workspace and explicit tuning protocol. Candidate pool/alignment/dedup/budget/numerics/steady timing and immutable aliased C0snapshot unchanged. This does not by itself establish that earlier Dense anomalies are solved; per-primary envelope analysis and exact Dense GPU capture follow closure.
- **Next:** Validate API/aliases/rebounds, capture Dense128GPU, reassess three-iteration outcomes and compiler/timeline/primary-source evidence before separately testing PLAN M2 stage contributions.
- **Archive:** trajectory/20261013_153023_iter-21/

Iter21 post-closure primary comparison (quick20/5/1, mixed historicalfull protocols, notacceptance): bf16 128^3 9.987us Dense/SW=0.420x priorbest/SW=0.413x; fp16 128^3 10.195us Dense/SW=0.397x priorbest/SW=0.426x; bf16 256^3 11.078us Dense/SW=0.478x priorbest/SW=0.573x; fp16 256^3 10.955us Dense/SW=0.441x priorbest/SW=0.585x; bf16 512^3 12.027us Dense/SW=0.506x priorbest/SW=0.914x; fp16 512^3 11.946us Dense/SW=0.513x priorbest/SW=0.928x; bf16 1024^3 20.034us Dense/SW=0.640x priorbest/SW=0.990x; fp16 1024^3 19.824us Dense/SW=0.512x priorbest/SW=0.991x. Configuration/TUNE evidence is now complete, numeric gatesremainfailed.

### Reassessment after iterations19–21 (before22)

Per-primary128/256/512/1024 prior-envelope latency did not improve by3% in three sourceiterations:19 compactBK64 seven-batch pipeline controls,20 rawcp.async doublebank16x16 owners,21 stronger setup-only smallLt tuning. Earlier16prefetch remains tiny/512 winner in diagnostic screens; Lt1 remains1024 winner. Inspect all per-shape entries; cross20/100-call historical measurements are not final acceptance. Full structural/API/alias/rebind tests through17 passed after21, and new51column records preserve actualboundDense/leaf/batch algorithm configuration. A single quick run with stronger tuning does not prove earlier Dense variation is eliminated.

Re-profiled exact Dense128 with the new singlePlan --profile-dense route: BF16GPU3.072us and FP16GPU3.008us, one warmed invocation each; selected IDs21, tile5/BF andtile11/FP, stage19, zero boundworkspace. The plan-wide maximumallocation is131072bytes and remains distinct. SQLite contains one GPUkernel and noGPUmemcpy/memset table; CUDA API includes cudaMemsetAsync but no captured device-memory activity. Captured hostAPI83/94us is inflated and notsteady submission overhead. Historical16SWGPU3.392us and bulkquick4.126us cannot be subtracted acrosscaptures to manufacture a breakdown. MaterialGPUwork remains; no physical launch/timer/bandwidthfloor established. Source661e878, binary/log identities and primaryrow configs inresults/small_reassessment_iter21.json. ncu is not retried after the original failure.

Compiler evidence:16registerprefetch has120regs vs15's96, actualnextLDG noearlyrawscoreboardwait until after7HMMA, but different cachepolicy;19H7/H8 have128/164regs, source-derived dynamic36/24KiB vsstatic1024byte resourcefields;20cpasync has145regs vs paired8's126 and16KiB compiledshared vs8KiB, withwait moved after3 of14HMMA. The last path therefore neither reduces physical registers nor overlaps the entireseven-productgroup. Allpaths passed their targeted threeCUDA12.8sanitizers. These observations constrain hypotheses, not dynamic occupancy attribution.

Re-read exact primaryCUDA12.4.1 [register pressure](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html#register-pressure), [occupancy](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html#occupancy), and [async copy](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory) sections from its official HTML. Async copies remove their intermediate global-to-shared registers, but this does not guarantee lower whole-kernel registers; higher occupancy alone does not guarantee speed. OfficialHTML snapshothash recorded inJSON; stalledwebfind cancelled, exactHTML retrieved onlogin with20secondbound.

Viable directions remain, so continue:22 PLAN6.5M2 fourlongC+onePscratch stagecontributions (explicit different FP32order, separateinteger/stagereference andunchangedoriginalFP64gate);23 newlyproposed8x8outputowners with bottom8Azero fornative16x8MMA (moretinyCTAs/fewerAcasts, doubledredundantTensorCorework, scratchCPUreviewpending). Neither is a promised improvement. Final frozenbinary launch/Graph full20warm/100calls/5groups/3fresh comparison remains required. No exhaustion or physicalfloor claim.

### Iter 22 — Accumulate per-stage Winograd contributions in four C fragments

- **Hypothesis:** PLAN M2 can lower physical accumulator register pressure while preserving FP32 S/T edges and testing its distinct per-stage FP32 order independently.
- **Changes:** Added Mega18 one-warp16x16 K16 stage products with four long C fragments plus one cleared P scratch, thirteen RN adds and one RN subtraction, general coefficients and alias-safe scalar output; separate integer/stage reference, CPU M2 model and Graph/rebind tests; small-suite future Mega19 mappings and combined CPU DAG/M2 verification frozen.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.426841 ms; mean speedup=0.3137x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_stage_m2 v18 | launch | 0.004165 | 0.005517 | 0.7549 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_stage_m2 v18 | launch | 0.004066 | 0.005838 | 0.6964 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_stage_m2 v18 | launch | 0.005235 | 0.008589 | 0.6095 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_stage_m2 v18 | launch | 0.004842 | 0.008758 | 0.5528 | 0.000737166 | 0 |
| 512,512,512 | bf16 | megakernel_stage_m2 v18 | launch | 0.006083 | 0.014752 | 0.4124 | 0.00598239 | 0 |
| 512,512,512 | fp16 | megakernel_stage_m2 v18 | launch | 0.006078 | 0.015091 | 0.4028 | 0.000753205 | 0 |
| 1024,1024,1024 | bf16 | megakernel_stage_m2 v18 | launch | 0.012930 | 0.045035 | 0.2871 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_stage_m2 v18 | launch | 0.010053 | 0.045242 | 0.2222 | 0.00075182 | 0 |
| 2048,2048,2048 | bf16 | megakernel_stage_m2 v18 | launch | 0.032077 | 0.338557 | 0.0947 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_stage_m2 v18 | launch | 0.032619 | 0.339779 | 0.0960 | 0.0007522 | 0 |
| 3072,3072,3072 | bf16 | megakernel_stage_m2 v18 | launch | 0.086331 | 1.022974 | 0.0844 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_stage_m2 v18 | launch | 0.088602 | 1.008022 | 0.0879 | 0.000752811 | 0 |
| 4096,4096,4096 | bf16 | megakernel_stage_m2 v18 | launch | 0.195440 | 2.346800 | 0.0833 | 0.00602273 | 0 |
| 4096,4096,4096 | fp16 | megakernel_stage_m2 v18 | launch | 0.206854 | 2.356114 | 0.0878 | 0.000753129 | 0 |
| 1024,512,1024 | bf16 | megakernel_stage_m2 v18 | launch | 0.009464 | 0.030414 | 0.3112 | 0.00599043 | 0 |
| 1024,512,1024 | fp16 | megakernel_stage_m2 v18 | launch | 0.009534 | 0.030726 | 0.3103 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_stage_m2 v18 | launch | 0.008253 | 0.030213 | 0.2732 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_stage_m2 v18 | launch | 0.008614 | 0.030714 | 0.2805 | 0.000751513 | 0 |

- **Analysis:** Quick20warm/20calls/5groups/1fresh completed18 finite rows, all frozen numerical gates fail exit2. Compiled stage kernel uses95 registers, source shared7168 bytes, zero ptxas stack/spills. Its128 BF5.5168us and FP5.84us are slower than prior prefetch16 screening; lowered register count alone does not establish speedup. Per-primary envelope analysis, GPU structure/Graph/sanitizers and compiled SASS audit follow closure.
- **Next:** Validate Mega18 with full structural/API tests and three standalone CUDA12.8 sanitizers, then integrate separately frozen Mega19 smaller8x8 owners and measure full same-binary launch/Graph candidates.
- **Archive:** trajectory/20261013_154902_iter-22/

Iter22 post-closure primary comparison (quick20/5/1, mixed historical full protocols, diagnostic only): BF16/FP16 stage18 latency at128 is5.5168/5.8384us versus prior observed4.1264/4.3392us;256 is8.5888/8.7584us versus6.3456/6.4064us;512 is14.7520/15.0912us versus10.9936/11.0864us;1024 is45.0352/45.2416us versus19.8298/19.6378us;2048 is338.5568/339.7792us versus73.0704/67.5408us. No primary shape gains3%. Physicalregister count falls from paired8's126 to95, but source-derived counts/occupancy are not a measured speed mechanism. All18 frozen numerical gates fail, finite; no Auto change. Root full structural/API/integer-stage/M1-bitwise-excluding18/Graph validation, isolated18 shortK/alias/rebind, and standaloneCUDA12.8memcheck/racecheck/synccheck all exit0. Exact measuredtest source snapshot is .scratch/test_correctness_iter22.cu fromdf323da; compiler/test evidence audit inprogress while additive23integration proceeds.

### Iter 23 — Split prefetched warp owners into eight-by-eight outputs

- **Hypothesis:** More logical owners can improve tiny-matrix SM coverage despite redundant lower-half native MMA work, with fewer raw registers and casts per owner.
- **Changes:** Added strict Mega19 native16x8MMA with bottom8A initialized zero and only top8x8 output, raw16u32 lookahead, seven complete P fragments and fixed U, current-pointer fallback19to16to15to14to12to7; isolated shortK/owner/signedzero/alias/rebind/Graph and M1-bitwise tests; strict31routes930; report envelope now chooses own minimum latency with same-row Dense ratio.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=1.037819 ms; mean speedup=0.3041x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.004115 | 0.004062 | 1.0130 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.004066 | 0.004198 | 0.9684 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.005288 | 0.006446 | 0.8203 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.004830 | 0.006485 | 0.7449 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.006122 | 0.015302 | 0.4000 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.006080 | 0.015269 | 0.3982 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.012810 | 0.109635 | 0.1168 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.010126 | 0.109402 | 0.0926 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.031363 | 0.744059 | 0.0422 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.033333 | 0.743530 | 0.0448 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.085571 | 2.447918 | 0.0350 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.088675 | 2.453768 | 0.0361 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.194704 | 5.880285 | 0.0331 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.206230 | 5.940480 | 0.0347 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.009110 | 0.049939 | 0.1824 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.009182 | 0.050101 | 0.1833 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.008032 | 0.049917 | 0.1609 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v19 | launch | 0.008325 | 0.049941 | 0.1667 | 0.00075151 | 0 |

- **Analysis:** Quick20warm/20calls/5groups/1fresh completed18 finite rows with all frozen numerical gatesfailed exit2. At128 the screen shows approximately4.06us BF16 and4.20us FP16, same-row Dense ratio near unity; physical96registers source5376shared and zero ptxas stack/spills. Grid doubles relative16 while aggregate nativeMMA/Bwork also doubles; full comparisons and GPU safety/compiler evidence follow closure. No threshold or Auto change.
- **Next:** Validate19 structure/bitwise/Graph and three standalone CUDA12.8 sanitizers, inspect native fragment/shared/prefetch SASS and warmed GPU128 timeline, then full same-binary40-label launch/Graph suite and31-route strict verification with final evidence/report.
- **Archive:** trajectory/20261013_160132_iter-23/
