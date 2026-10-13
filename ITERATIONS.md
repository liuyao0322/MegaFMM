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
