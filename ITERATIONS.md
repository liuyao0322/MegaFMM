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

| 24 | Construct native MMA operands directly in registers | 0.3401 | 0.617107 | screen-numerical-fail |

| 25 | Coalesce native MMA inputs with vector64 register routing | 0.3587 | 0.872000 | screened |

| 26 | Specialize original owner8 MMA at parent K128 | 0.3493 | 1.036923 | screened |

| 27 | Share transformed operands across cooperative M1 output warps | 0.3916 | 0.337076 | diagnostic |

| 28 | Fuse original inputs into uniform seven-product leaves | 0.3811 | 0.364023 | diagnostic |

| 29 | Matched packed-input seven-product leaves | 0.3329 | 0.288007 | measured-not-certified |

| 30 | Single-launch producer-consumer WGMMA pipeline | 0.4094 | 0.358867 | no-change |

| 31 | Single-launch TMA raw-alias two-bank WGMMA | 0.3970 | 0.292840 | regression |

| 32 | Overlap WGMMA consumer groups with vector producer | 0.5005 | 0.384269 | regression |

| 33 | Add third operand bank to vector wait1 pipeline | 0.4091 | 0.364720 | no-change |

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

Iter23 post-closure primary comparison (quick20/5/1, mixed historicalfull protocols, diagnostic only): bf16 128^3 4.0624us prior-envelope/SW=1.0158 Dense/SW=1.0130; fp16 128^3 4.1984us prior-envelope/SW=1.0335 Dense/SW=0.9684; bf16 256^3 6.4464us prior-envelope/SW=0.9844 Dense/SW=0.8203; fp16 256^3 6.4848us prior-envelope/SW=0.9879 Dense/SW=0.7449; bf16 512^3 15.3024us prior-envelope/SW=0.7184 Dense/SW=0.4000; fp16 512^3 15.2688us prior-envelope/SW=0.7261 Dense/SW=0.3982; bf16 1024^3 109.6352us prior-envelope/SW=0.1809 Dense/SW=0.1168; fp16 1024^3 109.4016us prior-envelope/SW=0.1795 Dense/SW=0.0926; bf16 2048^3 744.0592us prior-envelope/SW=0.0982 Dense/SW=0.0422; fp16 2048^3 743.5296us prior-envelope/SW=0.0908 Dense/SW=0.0448. 128FP crosses3% over prior observedSWenvelope;128BF improves about1.6%, otherprimarysizes do not gain. No independent performance or numerical acceptance. Lower96registers vs16's120 and moretinyowners coexist with redundant nativeMMA/Bwork; actual compiler/timeline evidence inprogress. Sourceea4dc50/binaryfrozen for final40-label/720-row fullsuite and31-route/930 strict checks, no source edits during measurement.

### Full small-matrix performance checkpoint after iteration23

This is a phase snapshot and not an optimization stop or completion of PLAN. The complete measurement source is `ea4dc50c57b08b4dee98c23c024a01bb94f836db`; later documentation/evidence commits do not replace that identity. Forty launch/Graph labels cover twenty routes, nine shapes and two dtypes, totaling720 checked SW rows. Every command uses20warmup/100calls/5alternatinggroups/3fresh correctness trials. All720 fixed numerical gates fail, all values are finite, with no unchecked or Dense fallback rows. Benchmark SHA256 is `a89d8cadf7f3dcbbecd41e6d7fac30de24ce55827bcbacf372126beaaf807d42`. Eighty before/after telemetry files identify the same H100 UUID; clocks are unlocked and telemetry does not establish within-group clock stability.

The observed envelope selects minimum SW latency per shape/dtype/mode, then reports that winner's same-row independently tuned ordinary cuBLASLt Dense comparison. It does not select the largest Dense/SW ratio. Times below are medians of five group means in microseconds, not single-call tails. Selecting on these measurements does not create an independently validated dispatch rule.

| Mode | dtype | square N | fastest observed SW | Dense us | SW us | same-row Dense/SW |
|---|---|---:|---|---:|---:|---:|
| launch | bf16 | 128 | Mega19 | 3.938 | 3.915 | 1.006 |
| launch | fp16 | 128 | Mega19 | 3.833 | 4.008 | 0.956 |
| launch | bf16 | 256 | Mega10 | 5.028 | 6.203 | 0.811 |
| launch | fp16 | 256 | Mega10 | 4.752 | 6.259 | 0.759 |
| launch | bf16 | 512 | Mega16 | 5.935 | 10.829 | 0.548 |
| launch | fp16 | 512 | Mega16 | 5.902 | 11.061 | 0.534 |
| launch | bf16 | 1024 | Lt1 | 12.364 | 19.643 | 0.629 |
| launch | fp16 | 1024 | Lt1 | 9.813 | 19.350 | 0.507 |
| Graph | bf16 | 128 | Mega19 | 3.483 | 3.629 | 0.960 |
| Graph | fp16 | 128 | Mega19 | 3.404 | 3.819 | 0.891 |
| Graph | bf16 | 256 | Mega16 | 4.470 | 5.844 | 0.765 |
| Graph | fp16 | 256 | Mega10 | 4.027 | 5.875 | 0.685 |
| Graph | bf16 | 512 | Hopper7 compactBK64 | 5.432 | 9.210 | 0.590 |
| Graph | fp16 | 512 | Hopper7 compactBK64 | 5.391 | 9.373 | 0.575 |
| Graph | bf16 | 1024 | Lt1 | 11.223 | 17.364 | 0.646 |
| Graph | fp16 | 1024 | Lt1 | 9.282 | 17.092 | 0.543 |

The BF128launch 0.6% observed gap is below the1.05 performance threshold and comparable to Dense's1.06% group CV (SW0.23%); it is not a stable accepted benefit. Historical originalLt0/SW=9.029 at that row is a separate cross-run implementation comparison, subject to clock/tuner/submission variation. The user's earlier128 Dense/SW≈0.90 refers specifically to iteration12 Mega10 BF128launch: same-row Dense4.17760014534us / SW4.64159995317us =0.900034511264, or11.10685% longer SW latency. That old quick20/5/1 numeric-failing observation must not be compared as if it shares the current full protocol.

Verification identity uses the same measurement source and test SHA256 `51791901d93f465db30d41f419f1bc2f7334b02936c9e8c0365e21130679d392`. Six labels complete: DAG, structural/API, full memcheck, racecheck smoke and synccheck smoke exit0; strict numerics exit2. CPU DAG3 and M2model6 checks PASS. Structural and memcheck each have1310CASE/147PASS and include shortK, Graph, rebinding and M1bits. Racecheck and synccheck each cover374candidate smoke CASE; zero errors/hazards/warnings do not imply those tools ran the extended suites. Standalone CUDA12.8 sanitizer2025.1.0.0 is used with unchanged CUDA12.4 application binaries. Strict31 requested configurations ×2dtypes×5patterns×3fresh =930 checks,558pass/372fail/nonfinite0. Mega18 M2 remains excluded from the M1bitset. Auto remains Dense and no gate is widened.

Evidence closure: `results/SMALL_REPORT.md` SHA256 `3bb95c7affae89f1503e4da92af7b788f06ce6b92d3e2682c1da161b42a29a3f`; `results/small_manifest.json` SHA256 `908a0dcbf02b0dc13e0eaead16e08e4a2644c1cedb72702c38a0f78eef58f904`; eight plots in both PNG/SVG; raw51-column CSVs,1440 matching TUNE configurations and2160 fresh CHECK metrics; all measurement-source file hashes match the declared commit. Full commands, actual bound algorithms, numerical failures, telemetry and coverage limits remain auditable in the report/manifest. Original large-matrix94-row/300-check history is retained unchanged.

Compiler/timeline audits remain separate from timing claims. Mega18 has95regs and no local spill, but its M2 RN contribution order is verified by SASS and is slower than the observed primary envelope. Mega19 has96regs, source/CUPTI5376B shared versus cuobjdump6400B, zero LOCAL/STACK or LDL/STL; seven of eight next-block loads precede the firstHMMA, with the eighth after it and consumption at the next stage. Single warmed128GPU captures are BF3.200us/FP3.232us; instrumented host APIs are inflated and cannot be subtracted from separate group measurements. Logical8x8 owners double aggregate nativeMMA/Bwork and add one-third original input bytes versus16x8; these are not measured DRAM traffic or floor evidence.

Next viable axis: direct native MMA register operands, separately testing16x8 and8x8 owners, removes shared S/T delivery/LDSM/barriers but increases source warp-load instructions (M16:8→32; M8:8→24), with cache-dependent sector repetition and uncompiled operand liveness. Frozen scratch prototype SHA256 `0e65d636194e501a94f55e713d1a5c37360a5331517a76494c91c4589fc41736` and independent15PASS CPU ABI/DAG/sector checks support an additive experiment, not a promised gain. First compile/resources/SASS, then M1 GPU bits and fixed numerical gates, structural/shortK/Graph/rebind/sanitizer validation and own-latency screens. Existing fastest paths remain available verbatim. No cap, physical floor or exhaustion condition is established; continue the authorized optimization loop.

### Iter 24 — Construct native MMA operands directly in registers

- **Hypothesis:** Removing shared delivery and warp barriers may reduce small-tile latency despite more global load instructions and operand register liveness.
- **Changes:** Added direct native M16/M8 controls Mega20/21 preserving FP32 DAG and complete M1 products, explicit variant mapping and current-binding fallback16to15to14to12to7; new shortK/random native bits/Graph/rebind/guard coverage and max21 policy validation; no gate or timing changes.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.617107 ms; mean speedup=0.3401x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.004178 | 0.004496 | 0.9292 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.004066 | 0.004666 | 0.8714 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.005294 | 0.007054 | 0.7505 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.004848 | 0.007066 | 0.6861 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.006141 | 0.012245 | 0.5015 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.006086 | 0.012256 | 0.4966 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.012883 | 0.059294 | 0.2173 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.010150 | 0.059230 | 0.1714 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.031315 | 0.438918 | 0.0713 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.032586 | 0.445530 | 0.0731 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.086395 | 1.497859 | 0.0577 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.089011 | 1.494301 | 0.0596 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.195075 | 3.453959 | 0.0565 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.206101 | 3.486920 | 0.0591 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.009170 | 0.030942 | 0.2963 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.009232 | 0.031176 | 0.2961 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.008048 | 0.030930 | 0.2602 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_direct_16x8 v20 | launch | 0.008346 | 0.031090 | 0.2684 | 0.00075151 | 0 |

- **Analysis:** Mega20 quick20warm20calls5groups1fresh completed18 finite rows with frozen numerical gates failed exit2. At128 BF4.496us FP4.6656us are slower than frozen source23 observed envelope. Compiler reports M16 REG128 and M8 REG120 with zero stack/spills and no source shared allocation; this is not a demonstrated occupancy mechanism. Cache and load granularity change versus15, while prefetch also changes versus16/19. M8 control and full GPU correctness/resources analysis follow closure.
- **Next:** Measure unchanged-binary Mega21 control, validate full M1 bits and shortK Graph rebinding, inspect generated LD widths/shared removal/P-U order, and run targeted standalone sanitizers; retain source23 snapshot and all earlier best paths.
- **Archive:** trajectory/20261013_171152_iter-24/

Iter24 post-closure checkpoint: measurement source `bb15c037009ab0441afe57d861454d0b74a57526`, benchmark SHA256 `a146d0b25a4d4403a28164dcab3ed60b8f76b71b05f47ec1559ce1d5ef8807e1`, test SHA256 `41efaa9304ba43fad312b298b5612100c6e1a306dfc35ee0470833e41925ea22`. The unchanged-binary Mega21 quick control completes18 finite numerical-failing rows, exit2, with BF/FP128latency4.6368/4.8560us and same-row Dense/SW0.899586/0.841516;256latency7.8672/8.0448us,51214.9520/15.2848us. Mega20 and21 have no primary ≥3% own-latency gain against the source23 observed envelope. This comparison mixes quick20/5/1 and prior full100/5/3 and is diagnostic, not a certified performance verdict.

Default structural/API validation exit0 has1458CASE/327PASS, including144 randomized native-fragment bitwise fixtures comparing20to15 and21to19 over two dtypes, three M/N layouts, K32/64/96, padded/packed strides, AB phase0/+16B and signed beta zero. All logical outputs are finite and byte-identical to the matching shared-operand M1 implementation; padding/trailing guards remain intact. Mega18 remains outside the complete-P M1 bitset. Six targeted standalone CUDA12.8 sanitizer commands (each tool ×20/21 with candidate filter) exit0, each with42CASE,72 corresponding native bitwise fixtures and6 Graph PASS records; memcheck/synccheck report zero errors and racecheck zero hazards/errors/warnings. These filtered commands additionally execute the corresponding shortK/Graph/rebind extensions, unlike source23's generic race/sync smoke suite. Structural equivalence does not establish the fixed random numerical gate. Strict33 requested configurations/25 actual route names produce990 gates/330 distributions,594pass/396fail/nonfinite0, exit2; per requested configuration18pass/12fail. Upper-range Dense fallback passes do not certify the native SW route.

Independent source/static compiler audit confirms four direct BF/FP native functions: Mega20 uses128 registers, Mega21 uses120, with SHARED/LOCAL/STACK0 and no LDL/STL spills. Each has7 static HMMA instructions in the K16 loop, seven independently zero-initialized P fragments, increasing-K self-accumulation, and fixed U only after the loop. M16 has16 emitted32-bit and16 U16 global loads per stage; M8 has8 emitted32-bit and16 U16 loads. The B pairs span different row-major K rows, requiring two scalar U16 loads and packing. The kernels emit no shared/LDSM/shuffle/barrier instructions. Same-binary15/19 also emit no BAR/WARPSYNC/MEMBAR, so source sync removal is not an emitted-barrier-instruction reduction. Compiler scheduling interleaves current-stage operands and early HMMA; all42 source operand sets are not claimed simultaneously live. Compared with15, cache/load granularity also changes; compared with16/19, next-stage lookahead is removed. More registers and unchanged useful bytes are observations, not a demonstrated occupancy or bandwidth limit.

Historical Graph coverage prose is corrected in `results/small_evidence_errata.md` while frozen source23 artifacts retain their original bytes and hashes. Identity19/20/21 captures alpha1/beta-0 with actual C0=nullptr; it refreshes D only before replay, with A/B unchanged. It validates null-C0 capture/replay and output guards, not mutated-A/B replay or active C0=D alias. Other nonzero-beta/fallback tests cover active alias. The next source iteration will refresh A/B at captured pointers and recompute original FP64 AB and, for M2, the independent stage reference on every replay. No timing data or numerical gate changes follow from this erratum.

Next viable operand-delivery experiment is coalesced vector64 raw input plus native register permutation, preserving full M1 P/U, tested separately for M16A64+B64 and M8A64+B64. Frozen scratch CUDA SHA256 `0e0b94d39ea4c6760a5f49aaed9fc27906720a5677ffaf1b375b3962cbfdb232` and independent actual-source CPU ABI/DAG/alignment audits PASS:6144 byte comparisons,4480 expression-tree comparisons,11520 boundary/alignment checks, plus8 frozen permutation tests. Explicit source schedules are12 global-vector loads/20SHFL/36 pair choices/8PRMT for M16 and8/16/24/8 for M8; they do not predict total issue, DRAM or time. M16 A phases can repeat requests for the same sector. This remains a viable measured experiment, not floor/exhaustion evidence or an optimization stop.

Independent24 evidence is frozen in `results/small_direct_profile_iter24.json`, SHA256 `fa652362742b093bc1d6d479fa9408323fcb55aaa3e82b1375b643209615d9b8` (103646 bytes). Its57 raw input hashes and2 binary hashes match before publication, and source23 report/manifest/profile retain their previously recorded SHA256 values. New-source iteration25 can now begin; its future strengthened Graph results must be attributed to that source, never retroactively to24.

### Iter 25 — Coalesce native MMA inputs with vector64 register routing

- **Hypothesis:** A64/B64 producer lanes and full-warp butterfly/PRMT can reduce explicit global-load operations while testing the added shuffle and register cost independently for M16 and M8 owners.
- **Changes:** Added independent Mega22 M16 and Mega23 M8 coalesced-register controls with full-P fixed-U M1, current-binding fallback through16/15/14/12/7, unchanged old CUDA functions, max23 policy, native-bitwise and guard coverage, and fresh original A/B Graph replay references.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.872000 ms; mean speedup=0.3587x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.005874 | 0.004653 | 1.2624 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.005746 | 0.004827 | 1.1903 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.005747 | 0.007179 | 0.8005 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.005675 | 0.007330 | 0.7743 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.006333 | 0.013909 | 0.4553 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.006286 | 0.013590 | 0.4626 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.013256 | 0.081770 | 0.1621 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.010243 | 0.081053 | 0.1264 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.032570 | 0.670130 | 0.0486 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.033386 | 0.653080 | 0.0511 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.086091 | 2.048141 | 0.0420 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.089293 | 2.040165 | 0.0438 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.195094 | 4.921531 | 0.0396 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.206064 | 4.982055 | 0.0414 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.011555 | 0.041798 | 0.2765 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.011619 | 0.041603 | 0.2793 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.008168 | 0.041629 | 0.1962 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_coalesced_16x8 v22 | launch | 0.008475 | 0.041550 | 0.2040 | 0.00075151 | 0 |

- **Analysis:** This is the predeclared nine-shape diagnostic20warm/20calls/5groups/1fresh launch screen of Mega22; default GPU structural/API/M1 bits and refreshed-input Graph tests passed before the screen. Fixed numerical gates and AutoDense remain unchanged. Own latency and prior-envelope comparison, Mega23 same-binary control, strict1050, six CUDA12.8 sanitizer runs, and compiled-resource audit follow this mandatory closure; no complete acceptance or floor is inferred from screening.
- **Next:** Validate the same frozen source25 binaries and Mega23 control, inspect SASS/resources, and run same-binary full launch/Graph comparisons if a primary own-latency improvement crosses3percent.
- **Archive:** trajectory/20261013_180222_iter-25/

Iter25 post-closure checkpoint: measurement source `9c9c188f566ed05c17fb30429308b65586879bcc`, benchmark SHA256 `2c302163cda0cfd2892b71492ef24b5fcf6e59b0179bd229cdd7d19ff655855b`, test SHA256 `2ecf29d619775d1a8153ac299e5817c982da4294f74b1e3750e949979d6f105e`. Same-binary Mega23 control completes18 finite numerical-failing rows exit2. Neither22 nor23 has a ≥3% primary own-SW-latency improvement against source23 observed minima. The predeclared expansion trigger is not met, so this iteration does not run a new full launch/Graph performance suite. The quick20warm20calls5groups1fresh versus prior full100calls5groups3fresh comparison is diagnostic and does not certify a causal regression or independent benefit.

| N | dtype | prior observed SW µs | Mega22 SW µs | Mega23 SW µs | Dense22 µs | Dense23 µs |
|---:|---|---:|---:|---:|---:|---:|
| 128 | bf16 | 3.9146 | 4.6528 | 4.1680 | 5.8736 | 5.4560 |
| 128 | fp16 | 4.0077 | 4.8272 | 4.3392 | 5.7456 | 5.3872 |
| 256 | bf16 | 6.2029 | 7.1792 | 6.9184 | 5.7472 | 5.4880 |
| 256 | fp16 | 6.2592 | 7.3296 | 6.9440 | 5.6752 | 5.5008 |
| 512 | bf16 | 10.8294 | 13.9088 | 15.7968 | 6.3328 | 6.1728 |
| 512 | fp16 | 11.0608 | 13.5904 | 15.5808 | 6.2864 | 6.1360 |
| 1024 | bf16 | 19.6432 | 81.7696 | 110.6464 | 13.2560 | 13.2880 |
| 1024 | fp16 | 19.3498 | 81.0528 | 110.2208 | 10.2432 | 10.3008 |

The apparent128 same-row Dense/SW>1 comes with a higher measured Dense latency, while both new SW controls themselves are slower than the source23 envelope. BF128 Lt algorithm fields remain id21/tile5/stages19; FP128 id21/tile11 selects stages13 rather than prior19. Exact candidate/workspace/alignment/tuning configuration is recorded in each CSV/TUNE log. GPU clocks were unlocked; the pre-control SM345MHz snapshot is not evidence of frequencies within timed groups or an explanation of the Dense shift. These ratios do not establish a new implementation benefit or complete the user's four-size goal.

Default structural/API validation exit0:1606CASE/507PASS,144 direct and144 new coalesced native bitwise fixtures,48 GraphPASS/144 replay. Every Graph replay refreshes rounded original A/B at fixed captured addresses outside capture on the same stream, zeros and recomputes the original-input FP64 AB reference, and independently recomputes Mega18's M2 stage reference. Logs explicitly record refresh_ab=1, actual alias and nullC0; this strengthened coverage belongs to25 and must not be attributed to23/24. Strict35requested configurations ×2dtype×5patterns×3fresh=1050 gates/350distributions,630pass/420fail(nonfinite0), exit2; uniform/range each account for210fail.27 actual route names are observed. Mega18 remains excluded from complete-P M1 bits. Numerical floors/conjunction and AutoDense stay unchanged; structural equivalence does not certify the numerical gate.

Six separate CUDA12.8 Compute Sanitizer2025.1.0.0 processes (each22/23 ×memcheck/racecheck/synccheck), running the unchanged CUDA12.4 application, all exit0. Each filtered run covers42CASE,72 respective randomized coalesced native bits and6 GraphPASS/18 refreshed-A/B replays as well as shortK/fallback/rebind checks. Memcheck/synccheck report0errors; racecheck0hazards/0errors/0warnings. No additional GPU process remains running.

Independent compiler audit: new BF/FP M16 uses96regs and M8 uses72, SHARED/LOCAL/STACK0 and no shared/LDSM/local-spill/barrier instructions. Each stage has7self-C HMMA,24/16final packed RN conversions,12/8LDG.E.64,20/16SHFL.BFLY and8PRMT. Full seven P accumulators survive the K16 loop; post-loop fixedU accounts for28/14FADD and8/4STG.E.64. Static totalinstructions440/344, compared with direct20'sFP472/BF480 and21's376. Old20/21 SASS instructions/encodings are unchanged. These static counts are not dynamic issue, DRAM bytes, achieved occupancy or a demonstrated latency cause. ncu is not retried under the session lock.

Evidence `results/small_coalesced_profile_iter25.json` binds the seven actual source hashes, both binary hashes, new raw25 logs/CSVs/exits, strict counts, sanitizer coverage, exact configurations and compiled functions to9c9c188 rather than later documentationHEAD. Source23 report/manifest and source24 profile remain byte-frozen. All36 new diagnostic rows fail the numerical gate; no unqualified speedup or floor/exhaustion/completion claim follows. Next authorized test axis is original19'sK128 fixedkh64 no-unroll versus full-unroll, with unchanged helpers/P/U and current-binding fallback. It will be a separate source26 iteration after this evidence checkpoint, preserving all previous fastest paths.

### Iter 26 — Specialize original owner8 MMA at parent K128

- **Hypothesis:** A fixed four-stage leaf bound, independently tested with no-unroll and full-unroll, may reduce loop-control and address scheduling cost while retaining original19 raw lookahead and complete-P fixed-U arithmetic.
- **Changes:** Added Mega24 fixedkh64 no-unroll and Mega25 fixedkh64 full-unroll, exactK128 plus original19guards, current-binding fallback19to16to15to14to12to7, immutable old CUDA functions and Lt metadata, max25 validation, K128 native bits and stage weights, refreshed-AB Graph and rebind, and60new strict candidate checks including requiredHighHalfDense fallback.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=1.036923 ms; mean speedup=0.3493x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_warp_mma_8x8_k128_no_unroll v24 | launch | 0.005638 | 0.004210 | 1.3394 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_warp_mma_8x8_k128_no_unroll v24 | launch | 0.005541 | 0.004344 | 1.2755 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.005584 | 0.006536 | 0.8543 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.005637 | 0.006557 | 0.8597 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.006346 | 0.015370 | 0.4129 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.006309 | 0.015402 | 0.4096 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.013246 | 0.108346 | 0.1223 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.010277 | 0.109470 | 0.0939 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.031629 | 0.743805 | 0.0425 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.032744 | 0.761750 | 0.0430 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.086381 | 2.464696 | 0.0350 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.089274 | 2.452762 | 0.0364 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.195453 | 5.918488 | 0.0330 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.206690 | 5.853033 | 0.0353 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.009141 | 0.049944 | 0.1830 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.009230 | 0.050082 | 0.1843 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.008002 | 0.049894 | 0.1604 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_warp_mma_8x8_prefetch v24 | launch | 0.008330 | 0.049930 | 0.1668 | 0.00075151 | 0 |

- **Analysis:** The predeclared diagnostic nine-shape20warm20calls5groups1fresh Mega24 launch screen completed18finite numerical-failing rows exit2 after default GPU structural/API/M1 native-bit and refreshed-input Graph validation exit0. OnlyK128rows use the new specialization; otherKrows retain the original19chain. Compiler/source evidence, independent Mega25 same-binary control, strict1110 and six targeted CUDA12.8 sanitizer checks follow mandatory closure; same-row Dense shifts and quick/full historical differences do not establish own-code gain.
- **Next:** Measure unchanged-binary Mega25, audit actualloop/unroll resources and lookahead, validate strict and targeted sanitizers, and expand same-binary original19/new24/new25 full launchGraph only if primary ownlatency improves3percent. Reassess three consecutive nonimproving delivery/control axes before source27.
- **Archive:** trajectory/20261013_184447_iter-26/

Iter26 post-closure checkpoint: the measurement source is `d5ea407e42b96fe8222a0a995591da9be36109b9`, benchmark SHA256 is `1e6e774426ecce21c74de1beaf68c401321f8477cd901ead9e24ed85f8497a2a`, and test SHA256 is `7797c64ed852035262e7394280672efdc55ccbfd5092c2f2c63e0ee5ce8fadb1`. The full build and default GPU validation exit0. Full ptxas output is retained. Independent CPU source review passed: all 21 older CUDA files are byte unchanged, and removing the new Plan branches/max25 restores the frozen source25 Plan exactly.

The same-binary Mega25 control completes 18 finite rows, all failing the numerical gate, with exit2. Each screen uses the new specialization in only two K128 rows; the other 16 rows execute the original19 fallback chain. Neither control produces a primary own-SW latency improvement of at least 3% against the observed source23 minimum, so the predeclared full performance expansion was not triggered. Current quick sampling (20 warmup / 20 calls / five groups / one fresh check) versus historical full sampling (20 / 100 / five / three) remains diagnostic, with unlocked-clock, selection and submission caveats.

| N | dtype | source23 observed SW µs | Mega24 SW µs | Mega25 SW µs | Dense24 µs | Dense25 µs |
|---:|---|---:|---:|---:|---:|---:|
| 128 | bf16 | 3.9146 | 4.2096 | 4.0016 | 5.6384 | 4.1920 |
| 128 | fp16 | 4.0077 | 4.3440 | 4.1344 | 5.5408 | 4.0800 |
| 256 | bf16 | 6.2029 | 6.5360 | 6.4928 | 5.5840 | 5.2928 |
| 256 | fp16 | 6.2592 | 6.5568 | 6.4720 | 5.6368 | 4.8448 |
| 512 | bf16 | 10.8294 | 15.3696 | 15.3248 | 6.3456 | 6.1120 |
| 512 | fp16 | 11.0608 | 15.4016 | 15.3296 | 6.3088 | 6.1008 |
| 1024 | bf16 | 19.6432 | 108.3456 | 109.2928 | 13.2464 | 12.9136 |
| 1024 | fp16 | 19.3498 | 109.4704 | 110.1472 | 10.2768 | 10.0736 |

All 36 diagnostic rows fail the frozen original-input FP64 gate, with nonfinite0. Only the 128 cubic rows are newly native. Mega25 BF128 has same-row Dense/SW ≈1.0476, below the 1.05 performance condition, and is slower than the historical own-SW minimum; FP128 Dense/SW is below1. The Dense24 screen is slower than prior Dense measurements. Exact selected Lt configurations and TUNE records are retained; the single SM345MHz snapshot does not represent clocks within timed groups. Numerical failures and the other primary sizes also prevent the active four-size goal. Auto, numerical floors/conjunction, tuning protocol and CSV schema remain unchanged.

Default validation reports 1754 CASE / 595 PASS, including 144 direct, 144 coalesced and 48 paired K128 native fixtures, plus 60 Graph PASS / 180 fresh-A/B replays. The new K128 fixtures compare24/25 with original19 on 32², 128² and 128×256 shapes, with stage weights, padding, A/B+16 and beta±0; finite-output, padding and trailing guards are checked. K96/K160 original19 fallback, input rebind at M256 N256 K128, D+4, odd output strides, dynamic coefficients, C0/D alias and restoration are covered. Full Graph shapes32/96 use K128; the partial 80×112×48 fallback is retained. Before every replay, rounded original A/B are refreshed at captured pointers outside capture on the same stream; the FP64 AB reference is cleared and recomputed, and Mega18's stage reference is independently recomputed. Mega18 M2 remains excluded from complete-P M1 bitwise comparisons.

Strict validation produces 1110 gates / 370 distributions: 666 pass, 444 fail, nonfinite0, exit2. The original 35 requested configurations at K256³ retain 1050 gates (630 pass / 420 fail). New24/25 at M256 N256 K128 add60 (36 pass / 24 fail), comprising 48 native checks and 12 required HighHalf Dense fallbacks. Actual-route assertions check the certified new kernels and HighHalf Dense. Uniform/range each account for222 failures. The five patterns, three trials and fixed gate are retained. Untuned Dense in the new numerical-only group is not a performance baseline.

Six separate CUDA12.8 Compute Sanitizer2025.1.0.0 processes run the unchanged CUDA12.4 application and all exit0. Each variant/tool filter reports 42 CASE / 43 PASS, 24 respective paired K128 native fixtures, six Graph PASS / 18 fresh-A/B replays, plus short-K fallback and input/output rebind. Memcheck and synccheck report0 errors; racecheck reports0 hazards,0 errors and0 warnings. All GPU processes are finished.

Independent compiled audit passed. Original19 uses REG96 / 440 static instructions / seven HMMA; fixed no-unroll24 uses REG94 / 416 / seven HMMA; full-unroll25 uses REG86 / 760 / 28 HMMA. FP/BF counts agree. STACK/LOCAL and ptxas stack/spill stores/spill loads are zero. ptxas shared memory5376B and ELF SHARED6400B are recorded as separate fields; no CUDA function attribute was queried. Variant24 has a real `0x1600 @!P4 BRA 0xc10` backedge, and next≥64 suppresses all eight lookahead LDG instructions. Variant25 has no K loop, 32 LDG representing four raw banks, and seven persistent C chains across four ordered HMMA stages. Both new functions complete all seven P products before 14 FADD instructions for fixed U and four STG.E.64 output instructions. Original19 SASS address/instruction sequences match the source25 binary exactly. Variant24's eighth lookahead load follows its first HMMA, and variant25 bank loads interleave product HMMA; source lookahead placement alone does not prove load/compute overlap. Static counts include NOP and terminal self-BRA. Both controls execute 28 active HMMA and 32 active raw64bit loads at K128. Lower register counts and reduced loop control do not establish an occupancy or latency mechanism.

Reassessment of iterations24–26 is complete before27: direct native ABI, coalesced vector/permutation and fixed K128 loop controls each lack a primary own-SW screen improvement of at least3%. [CUDA12.4.1 pragma unroll](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#pragma-unroll) and the [pinned CuTe config](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cute/config.hpp) support the intended no-unroll/full-unroll controls; actual SASS confirms backend realization. [PTX8.4 native MMA/cache operators](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html) and [pinned SM80 atoms](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cute/arch/mma_sm80.hpp) constrain operand ABI and typed accumulation. Cache hints do not establish measured DRAM traffic. The original `results/profile_baseline.txt` records `/tmp/nsight-compute-lock` failure, InterprocessLockFailed, exit9 and no profiled kernels; the session prohibits retry and none was attempted. Historical single-warmed Mega19 GPU durations3.200/3.232µs and Dense128 GPU durations3.072/3.008µs remain separate captures and cannot establish a launch floor by subtraction.

The new profile and `results/small_reassessment_iter26.json` bind actual d5ea407 source, binaries, raw configurations, logs and exits, plus prior frozen artifacts. Source23/24/25 results remain unchanged; later documentation HEAD is not the measurement identity. A viable next axis for512/1024 is fused original-input transforms and batched Tensor Core products followed by the existing fixed-U combine: two launches instead of three, with no global packed A/B storage. Original final RN casts, full-P arithmetic, the numerical gate and caller-owned workspace remain requirements. Root-owned scratch needs a separate27 lease and is not part of26. Further geometry, reuse and stage-specific address propagation for128/256 remain viable. No hardware floor, exhaustion or goal completion is established; historical faster paths are retained and optimization continues.

### Iter 27 — Share transformed operands across cooperative M1 output warps

- **Hypothesis:** A cooperative CTA can reuse each transformed A/B tile across output warps and reduce repeated S/T work at512/1024 while preserving complete seven-P M1 arithmetic.
- **Changes:** Add Mega26 cooperative32x16 and Mega27 cooperative32x32, strict native metadata and current AB16 guards, packed7/commonDense fallback, max27, fresh-input Graph/native-bitwise/alias/stride/rebind tests; preserve all22 old CUDA implementations, numerical gate, Lt tuner and51column schema.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.337076 ms; mean speedup=0.3916x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.005627 | 0.005590 | 1.0066 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.005501 | 0.005723 | 0.9611 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.005621 | 0.008262 | 0.6803 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.005642 | 0.008443 | 0.6682 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.006326 | 0.013349 | 0.4739 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.006368 | 0.013507 | 0.4715 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.013195 | 0.039114 | 0.3374 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.010318 | 0.038310 | 0.2693 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.031979 | 0.239938 | 0.1333 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.033104 | 0.242261 | 0.1366 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.086795 | 0.804962 | 0.1078 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.089411 | 0.813312 | 0.1099 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.195350 | 1.862882 | 0.1049 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.206323 | 1.863450 | 0.1107 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.011352 | 0.026766 | 0.4241 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.011515 | 0.027221 | 0.4230 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_cooperative_32x16 v26 | launch | 0.008374 | 0.027070 | 0.3094 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_cooperative_32x16 v26 | launch | 0.008736 | 0.027203 | 0.3211 | 0.00075151 | 0 |

- **Analysis:** Predeclared nine-shape two-dtype Mega26 diagnostic20warm20calls5groups1fresh completed18finite numerical-failing rows exit2 after defaultGPU1902CASE667PASS with36newnativepairedbitwise and72Graph216freshABreplays. Core/test independentCPUauditPASS. Native complete-P stage sharing compiles REG126/128 with no spill; actual latency and own-primary signal will be evaluated from row medians after mandatory closure. Same-row Dense changes cannot establish code gain.
- **Next:** Measure same-binary Mega27 control; validate strict1170 and six isolatedCUDA12.8 sanitizer runs; audit compiled CTA barriers/load roles/LDSM/fullP U/resources; expand new versus old-best same-binary full launch/Graph only if primary ownSWlatency improves at least3percent.
- **Archive:** trajectory/20261013_193314_iter-27/

Iter27 post-closure checkpoint: the measurement source is `cad199f7a9eb35b3c5154be95f803f4d04257709`. Benchmark SHA256 is `5340abfb308b386301af11bf34000e928ea01456553ad39a865de2b0aafacd3f`; test SHA256 is `7da56b6b1a80c72d4b98b309ced2c10e3d35ed6eb44e82d6d0b8c5a34ee8e80e`. The full build completed exit0 and retains ptxas output. All22 older CUDA files are byte unchanged. The frozen prototype device body, CuTe output ownership, producer coverage, vector endpoints, fixed S/T/P/U order and four Plan dispatch points passed independent CPU review. Removing the four new Plan branches and restoring max25 recovers the source26 Plan byte exactly.

The same-binary Mega27 control also completes18 finite rows, all failing the frozen numerical gate, exit2. Both screens use the respective native cooperative route in all18 rows, rather than measuring a fallback. The diagnostic protocol is20 warmup /20 calls per group /five groups /one fresh check; historical source23 full sampling is20/100/five/three. Clocks are unlocked. No primary own-SW latency improves by at least3% against the observed source23 minimum, so the predeclared full performance expansion was not triggered. Same-row Dense variation is not an implementation gain.

| N | dtype | source23 observed SW µs | Mega26 SW µs | Mega27 SW µs | Dense26 µs | Dense27 µs |
|---:|---|---:|---:|---:|---:|---:|
| 128 | bf16 | 3.9146 | 5.5904 | 5.8304 | 5.6272 | 4.1744 |
| 128 | fp16 | 4.0077 | 5.7232 | 5.9584 | 5.5008 | 4.0656 |
| 256 | bf16 | 6.2029 | 8.2624 | 8.1776 | 5.6208 | 5.2480 |
| 256 | fp16 | 6.2592 | 8.4432 | 8.2576 | 5.6416 | 4.8272 |
| 512 | bf16 | 10.8294 | 13.3488 | 12.6160 | 6.3264 | 6.1264 |
| 512 | fp16 | 11.0608 | 13.5072 | 12.6704 | 6.3680 | 6.0912 |
| 1024 | bf16 | 19.6432 | 39.1136 | 31.4480 | 13.1952 | 12.8848 |
| 1024 | fp16 | 19.3498 | 38.3104 | 30.6704 | 10.3184 | 10.1984 |

Default GPU validation exit0:1902 CASE /667 PASS, including144 direct,144 coalesced,48 K128 and36 new cooperative paired native bitwise fixtures (372 total), plus72 Graph PASS /216 fresh-A/B replays. New26/27 native fixtures compare with packed7 on shortK32/64/96 and minimum/rectangular/512 shapes, with input+16, C0/D+4, padded input, odd output strides, general coefficients, legal C0=D and beta±0/nullC0. New exact fixtures distinguish26 native versus27 packed fallback at M64 N32, cover complete M64 N64,512 outputs and partial M80. Each new Graph uses active beta/C0=D and odd output strides, refreshes rounded original A/B at fixed captured addresses outside capture on the same stream, clears and recomputes original-input FP64 AB, and restores C0 before replay. Old identity/K128 Graph scope is unchanged; Mega18 M2 remains excluded from the complete-P M1 bitwise set.

Strict validation exit2:1170 gates /390 distributions,702 pass /468 fail, nonfinite0. Original35 configurations at K256³ retain1050 gates (630/420); K12824/25 retain60 (36/24); cooperative26/27 add60 (36/24). Each new two-control group contains48 native certified-range checks and12 required HighHalf Dense fallbacks. Uniform and range each account for234 total failures. Original fixtures, seeds, numerical floors/conjunction and the Lt tuning/51-column protocol remain unchanged. All36 diagnostic performance rows fail the frozen benchmark numerical gate; Auto stays Dense.

Six separate CUDA12.8 Compute Sanitizer2025.1.0.0 processes run the unchanged CUDA12.4 application. Each26/27 ×memcheck/racecheck/synccheck process exits0, with42 CASE /37 PASS,18 paired native fixtures and six Graph PASS /18 fresh-A/B replays. Memcheck reports0 leaked bytes and0 errors; racecheck0 hazards/0 errors/0 warnings; synccheck0 errors. All GPU processes have finished.

Independent compiler audit passed for all four instances. Mega26 usesREG126, ptxas smem10752B, ELF SHARED11776B and984 static instructions; Mega27 usesREG128, ptxas14336B, ELF15360B and976 static instructions. STACK/LOCAL and ptxas stack/spill stores/spill loads are zero. No CUDA function attribute was queried, so shared fields remain separate. Variant26 has four128-bit A loads and four64-bit B loads per thread per stage;27 has mutually exclusive low64 A/high64 B producers with four128-bit loads per producer. All14 LDSM and14 HMMA per warp/K16 stage are unpredicated. Both CTA barriers are unpredicated and encompass all shared reads. The compiler places the second barrier before the final eight HMMA, after all operands have reached registers; this is safe, but it is not a compiled barrier after all MMA completion. Seven independent P chains retain ascending complete-K accumulation before fixed U. Both beta paths keep separate alpha FMUL; active beta uses alpha products as FFMA addends, with four C0 loads before four corresponding stores for each output group. Static60 FMUL /64 STG include duplicated beta paths; one actual thread path executes32 FMUL /32 STG, plus32 FFMA only for active beta. Static instructions, register counts, shared fields and source reuse do not establish dynamic issue, DRAM traffic, achieved occupancy or a measured bottleneck. ncu was not retried.

`results/small_cooperative_profile_iter27.json` binds actual cad199f source, both binaries, all new raw logs/CSVs/exits, exact Lt configurations, compiled resources and older frozen evidence. The source23 complete performance envelope remains unchanged, including its sampling/selection limits and numerical failures. Source23/24/25/26 artifacts are byte frozen; later documentation HEAD is not the measurement identity. Cooperative sharing is a distinct tested mechanism after the24–26 reassessment, with no new primary own-latency win. Existing faster paths remain available. Root-owned fused original-input transform plus batched leaf with the existing complete-P/fixed-U combine remains a viable separate next mechanism, subject to a new formal lease. The user's four-size goal remains active and unmet; no hardware floor, exhaustion or optimization completion is established.

### Iter 28 — Fuse original inputs into uniform seven-product leaves

- **Hypothesis:** One full-P accumulator per CTA reduces register pressure while sharing each transformed operand across output warps.
- **Changes:** Add Hopper9/10 BM32 BN32/64 BK32 fused original-input leaves, compact P-only workspace, current-binding zero-scratch Dense fallback, and required existing combine as the second launch. Preserve all older kernel TUs, numerical gates, Auto and Lt metadata.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.364023 ms; mean speedup=0.3811x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.005968 | 0.007314 | 0.8160 | 0.00605216 | 0 |
| 128,128,128 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.005910 | 0.007528 | 0.7851 | 0.000744325 | 0 |
| 256,256,256 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.005989 | 0.008845 | 0.6771 | 0.00597914 | 0 |
| 256,256,256 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.007662 | 0.009152 | 0.8372 | 0.000737165 | 0 |
| 512,512,512 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.006502 | 0.013608 | 0.4778 | 0.00598238 | 0 |
| 512,512,512 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.006362 | 0.013598 | 0.4678 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.013422 | 0.043528 | 0.3084 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.010245 | 0.043061 | 0.2379 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.032155 | 0.268819 | 0.1196 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.032546 | 0.272982 | 0.1192 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.086990 | 0.861043 | 0.1010 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.089555 | 0.850288 | 0.1053 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.195718 | 2.001274 | 0.0978 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.206733 | 2.050360 | 0.1008 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.011802 | 0.025312 | 0.4662 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.011677 | 0.025373 | 0.4602 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | winograd_fused_input_32x32 v9 | launch | 0.008405 | 0.025101 | 0.3348 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | winograd_fused_input_32x32 v9 | launch | 0.008781 | 0.025235 | 0.3480 | 0.00075151 | 0 |

- **Analysis:** Nine-shape both-dtype end-to-end launch quick screen uses20warm/20calls/5groups/1fresh and same-row independently tuned Dense. Raw numerical failures remain visible. Default structural tests, four direct seven-P integer fixtures,36 random final-M1 pairs and fresh-A/B Graph tests passed before screening. Own-latency comparisons and compiled-resource interpretations follow in a separate evidence checkpoint; this quick screen does not certify performance or numerics.
- **Next:** Measure Hopper10 control; complete mandatory same-binary1024 full Hopper9/Hopper10/Lt1/H7 in launch and Graph20warm100calls5groups3fresh even without quick gain, then original strict gate, six CUDA12.8 sanitizers and independent raw/compiler audit.
- **Archive:** trajectory/20261013_203807_iter-28/


### Iter28 evidence checkpoint — complete two-route1024 matrix

Measured source `68035fa6b35bebdbef660741cba513ed3e183fd1` adds Hopper9/10: one CTA-uniform seven-product fused-input leaf launch writes full-K FP32 P, followed by the required existing combine launch. Every run recomputes its original-input transforms. Native plans reserve P only and use zero-scratch Dense for incompatible rebindings; unsupported metadata/range/budget selects ordinary GEMM. All23 older CUDA TUs and frozen source23–27 evidence remain unchanged.

The nine-shape both-dtype quick screens retain20warm/20calls/5groups/1fresh. Neither new configuration improves a primary own-SW latency by at least3% against the retained historical envelope. The user explicitly requested actual1024 comparison of both routes, so same-binary full1024 was required despite that screening result: Hopper9/Hopper10/Lt1/H7, both dtypes and launch/Graph,20warm/100calls/5groups/3fresh,32heuristic candidates and the same8GiB cap. All input transforms/packing, leaf products, P writes/reads, fixed U and alpha/beta are included; no input-dependent preprocessing is cached.

| Mode | dtype | route | Dense us | SW us | Dense/SW | SW p10/p90 us | SW CV | gate |
|---|---|---|---:|---:|---:|---|---:|---|
| launch | bf16 | winograd_fused_input_32x32 v9 | 12.3837 | 43.1270 | 0.2871 | 43.0868/43.1683 | 0.000859 | FAIL |
| launch | fp16 | winograd_fused_input_32x32 v9 | 9.9226 | 42.9302 | 0.2311 | 42.9039/43.0237 | 0.001266 | FAIL |
| graph | bf16 | winograd_fused_input_32x32 v9 | 11.2976 | 40.9485 | 0.2759 | 40.9297/40.9761 | 0.000563 | FAIL |
| graph | fp16 | winograd_fused_input_32x32 v9 | 9.3990 | 41.1693 | 0.2283 | 41.1480/41.2093 | 0.000759 | FAIL |
| launch | bf16 | winograd_fused_input_32x64 v10 | 12.3357 | 38.1146 | 0.3236 | 38.0742/38.1633 | 0.001036 | FAIL |
| launch | fp16 | winograd_fused_input_32x64 v10 | 9.9872 | 38.4778 | 0.2596 | 38.4567/38.5236 | 0.000795 | FAIL |
| graph | bf16 | winograd_fused_input_32x64 v10 | 11.2298 | 36.2998 | 0.3094 | 36.2724/36.3217 | 0.000623 | FAIL |
| graph | fp16 | winograd_fused_input_32x64 v10 | 9.4278 | 36.7155 | 0.2568 | 36.6911/36.7302 | 0.000481 | FAIL |
| launch | bf16 | winograd_lt_batched v1 | 12.2349 | 19.2918 | 0.6342 | 19.2653/19.2972 | 0.000761 | FAIL |
| launch | fp16 | winograd_lt_batched v1 | 9.9318 | 19.6413 | 0.5057 | 19.6324/19.6522 | 0.000471 | FAIL |
| graph | bf16 | winograd_lt_batched v1 | 11.3667 | 16.9962 | 0.6688 | 16.9948/17.0018 | 0.000192 | FAIL |
| graph | fp16 | winograd_lt_batched v1 | 9.3722 | 17.3923 | 0.5389 | 17.3836/17.4536 | 0.002366 | FAIL |
| launch | bf16 | winograd_sm80_batched_k64 v7 | 12.2912 | 21.8685 | 0.5621 | 21.8548/21.9129 | 0.001364 | FAIL |
| launch | fp16 | winograd_sm80_batched_k64 v7 | 9.9328 | 22.3366 | 0.4447 | 22.2934/22.3381 | 0.000965 | FAIL |
| graph | bf16 | winograd_sm80_batched_k64 v7 | 11.2925 | 19.1539 | 0.5896 | 19.1300/19.1677 | 0.000879 | FAIL |
| graph | fp16 | winograd_sm80_batched_k64 v7 | 9.4995 | 19.7997 | 0.4798 | 19.7795/19.8452 | 0.001476 | FAIL |

All16 full rows and all48 original-input FP64 fresh gates fail the unchanged numerical threshold; nonfinite outputs are zero and Auto stays Dense. Packed-once Lt1 is the observed own-SW minimum among these four controls for each mode/dtype. Its same-row Dense/SW ratios remain below1. This result compares current backend/config choices; it is not a matched leaf experiment isolating only the placement of S/T. Row ordering and unlocked clocks limit causal attribution; p10/p90/CV describe five group means, not individual-call tails or confidence intervals.

At1024, fused native P payload and caller-reserved workspace are7MiB. Packed Lt1/H7 have7MiB transformed lowp inputs plus7MiB P (14MiB payload), while their actual caller-reserved workspace is46MiB, including32MiB reserved for Dense-fallback candidates. The tuned Lt1 batch used bound_workspace0 and H7 leaf has zero backend scratch. The table reports complete end-to-end time, not leaf-only time.

Default structural validation exits0 with1956CASE/731PASS, four direct seven-P integer fixtures,36 random final-D bitwise pairs against Mega7, and80Graph checks/240fresh-A/B replays. Strict validation exits2 with1230gates=738PASS+492FAIL,410distributions and NF0; its groups are1050old+60K128+60cooperative+60fused. New fused60 splits48native and12required HighHalf Dense. Final six CUDA12.8 compute-sanitizer2025.1.0.0 checks all exit0 with zero leaks/errors/hazards/warnings; each H9 check has32CASE/33PASS and H10 has34CASE/33PASS, two direct-P fixtures,18 final-D pairs and fourGraph checks/12freshAB replays. The initial H9 memcheck briefly overlapped the still-pending strict process, was retained separately as diagnostic_initial, and is excluded from the final six; its replacement ran independently after the other five. All timing completed before this diagnostic overlap.

Compiled BN32/64 instances use56/64 registers with no stack/local/spill. ptxas reports4096/6144B shared; ELF reports5120/7168B separately, and runtime function attributes were not queried. Static1744/1816 instruction totals include all seven product branches and padding. A CTA selects one product, so static28HMMA/14BAR totals are not its dynamic stage counts. No ncu retry, occupancy claim, physical-floor conclusion or four-size completion follows. Fused-specific tests do not add transpose, isolated ldb misalignment or32768 GPU boundaries; common overlap/overflow negative tests remain the earlier forced-Lt coverage.

Binary SHA256: benchmark `5fc2afccc67ee63afb0e7bfc68435e855909f4447f00a9739fa36a15e643193f`; tests `e34035a28da7bfb0fefb055f1b632b4089aa8e50e0807e302ef2b87c84640a54`. Source/binary/raw/config bindings and all16 full rows are in [results/small_fused_profile_iter28.json](results/small_fused_profile_iter28.json); the original nine-shape screen is archived under `trajectory/20261013_203807_iter-28/`. The active128/256/512/1024 goal remains unmet. Later evidence-only HEADs do not redefine the measured source commit.

### Iter 29 — Matched packed-input seven-product leaves

- **Hypothesis:** Construct S/T once in a fresh uniform pack and reuse the fused controls tile, K32 consumer and complete-P order; compare end-to-end input placement at1024.
- **Changes:** Add Hopper11/12 compact-input32x32/32x64 leaves and current-binding guarded three-launch pack/leaf/combine Plan routes; retain all24 old CUDA TUs, frozen numerical/Auto/Lt contracts, seven-P matched bitwise tests and mandatory24-row full1024 declaration.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.288007 ms; mean speedup=0.3329x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.005677 | 0.010038 | 0.5655 | 0.00605216 | 0 |
| 128,128,128 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.005590 | 0.010344 | 0.5404 | 0.000744325 | 0 |
| 256,256,256 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.005611 | 0.011266 | 0.4981 | 0.00597914 | 0 |
| 256,256,256 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.005602 | 0.011349 | 0.4936 | 0.000737165 | 0 |
| 512,512,512 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.006328 | 0.014720 | 0.4299 | 0.00598238 | 0 |
| 512,512,512 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.006314 | 0.014691 | 0.4298 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.013384 | 0.040662 | 0.3291 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.010358 | 0.040525 | 0.2556 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.031800 | 0.222539 | 0.1429 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.033211 | 0.225966 | 0.1470 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.086600 | 0.684434 | 0.1265 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.089246 | 0.677424 | 0.1317 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.195766 | 1.561149 | 0.1254 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.206790 | 1.561870 | 0.1324 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.011376 | 0.024158 | 0.4709 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.011426 | 0.024494 | 0.4665 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | winograd_packed_input_32x32 v11 | launch | 0.008403 | 0.024208 | 0.3471 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | winograd_packed_input_32x32 v11 | launch | 0.008752 | 0.024293 | 0.3603 | 0.00075151 | 0 |

- **Analysis:** Quick nine-shape two-dtype launch screen uses20warm/20calls/5groups/1fresh,32Lt candidates and8GiB shared cap. Results and original numerical failures are recorded exactly; this diagnostic does not establish certification. Same-binary second control, mandatory user full1024 six routes in both modes/dtypes, strict numerics, six CUDA12.8 sanitizers and resource audit follow before evidence checkpoint. Judge gains from own SW latency, not Dense denominator changes.
- **Next:** Measure Hopper12 control and full1024 H9/H10/H11/H12/Lt1/H7 launch/Graph BF16/FP16 at20warm100calls5groups3fresh, then close strict/sanitizer/compiler evidence without changing frozen source.
- **Archive:** trajectory/20261013_211356_iter-29/

## Iteration29: matched packed-input versus fused-input1024 controls

Measured source is `bdbcdd9191072014aff7fa99dd9e19afd026d55e`, after source28 evidence checkpoint `f08c5b3406b8d4727bfaa2f30092d437bdb82f01`. Hopper11/12 add a matched compact-input leaf with the frozen fused9/10 tile/shared/K32/MMA/full-P order. Every complete native invocation freshly packs original rounded A/B, runs the compact leaf and combines P: three launches. Original-input9/10 remain two-launch leaf+combine routes. All24 older CUDA translation units and historical23–28 evidence are byte-preserved. Policy maxH12/invalid13, actual original/compact binding guards, alias/cap/empty/scale handling, Lt tuner/config API and51-column schema retain the original contracts; Auto stays Dense.

The user explicitly required a same-source/binary full1024 comparison even without a primary >=3% quick-screen signal. All six requested routes below were measured with20warm/100calls/5groups/3fresh in both dtypes and launch/Graph, with heuristic_candidates32 as the query budget and8GiB shared workspace cap. Transformations, packing, seven full-K FP32 products, P traffic and fixed U/alpha/beta are included; no input preprocessing cache. The complete24-row table with both Dense/SW p10/p90/CV and workspace is [results/1024_TWO_ROUTES_REPORT.md](results/1024_TWO_ROUTES_REPORT.md); exact original rows/configs and raw hashes are [results/small_matched_measurements_iter29.json](results/small_matched_measurements_iter29.json).

|1024 ordinary launch route|dtype|same-row Dense µs|complete SW µs|Dense/SW|SW CV %|frozen numerical gate|
|---|---|---:|---:|---:|---:|---|
|winograd_fused_input_32x32 v9|bf16|12.3238|43.0096|0.286537|0.2506|FAIL, finite|
|winograd_fused_input_32x32 v9|fp16|9.9984|43.0349|0.232332|0.1188|FAIL, finite|
|winograd_fused_input_32x64 v10|bf16|12.2589|38.2541|0.320459|0.1149|FAIL, finite|
|winograd_fused_input_32x64 v10|fp16|9.9590|38.5088|0.258617|0.0513|FAIL, finite|
|winograd_packed_input_32x32 v11|bf16|12.2022|40.0851|0.304408|0.0984|FAIL, finite|
|winograd_packed_input_32x32 v11|fp16|9.9734|40.3974|0.246883|0.0355|FAIL, finite|
|winograd_packed_input_32x64 v12|bf16|12.2992|34.1165|0.360506|0.1531|FAIL, finite|
|winograd_packed_input_32x64 v12|fp16|9.9430|34.4358|0.288741|0.1442|FAIL, finite|
|winograd_lt_batched v1|bf16|12.2298|19.3750|0.631212|0.0608|FAIL, finite|
|winograd_lt_batched v1|fp16|9.9302|19.7350|0.503178|0.2337|FAIL, finite|
|winograd_sm80_batched_k64 v7|bf16|12.2003|21.8893|0.557365|0.0955|FAIL, finite|
|winograd_sm80_batched_k64 v7|fp16|9.9338|22.3165|0.445131|0.1141|FAIL, finite|

Within these six candidates, packed-once Lt1 has the smallest own SW latency for all four mode/dtype pairs; this observed selection is not a fixed dispatcher or certification. Ordinary BF16/FP16 Lt1 is19.3750/19.7350µs versus same-row Dense12.2298/9.9302µs (0.631212/0.503178×). Graph Lt1 is17.0726/17.8051µs versus11.3597/9.4397µs (0.665373/0.530167×). All24 full rows and72 fresh original-input FP64 checks fail the unchanged numerical gates withNF0. Dense speed variations do not establish code improvements.

The matched32×64 H12 run is34.1165/34.4358µs in ordinary BF16/FP16 versus its same-tile fused H10 at38.2541/38.5088µs: observed own-SW decreases10.8161%/10.5767%. Graph decreases13.7719%/13.1142%. Matched32×32 H11 ordinary decreases6.7996%/6.1286% versus H9. These are input-placement controls with accompanying compiled load/resource differences, not a pure DRAM/cache experiment or a bottleneck proof. Neither new control improves the retained overall primary own-latency envelope by>=3%; there is no automatic broader full-suite trigger. The full1024 measurements were still mandatory under the user's explicit request.

The H12/FP16/Graph same-row Dense group means contain an outlier: Dense CV59.0005% and p90=21.4297µs, while SW CV0.1246%. The original groups and0.299958 median ratio are retained without replacement or selecting a best rerun. That denominator does not establish a code gain; the input-placement reductions use SW's own latency. Timings used unlocked GPU clocks, and group quantiles/CV are not confidence intervals.

At1024, fused9/10 P payload/reserved is7MiB. Packed11/12/Lt1/H7 have lowp7A+7B=7MiB and P=7MiB (14MiB payload); actual caller-reserved46MiB includes32MiB maximum scratch for the Dense fallback candidate pool. The custom packed leaves need0 backend scratch; the chosen Lt1 batch also reports bound_workspace0. Maximum caller allocation and selected Lt algorithm scratch are distinct quantities.

Compiled matched BN32 uses40REG,16B stack and80B spill stores/loads, even though ELF LOCAL reports0; its SASS contains actual LDL/STL. BN64 uses48REG and0 stack/spill. Ptxas shared4096/6144B and ELF SHARED5120/7168B are separate tool fields; no runtime allocation attribute was queried. Four whole functions have1432/1456 static instructions across seven CTA-uniform product branches. Totals14LDG.E.128/14STS.128/28LDSM/28HMMA/14BAR/56scalarSTG are whole-switch static counts, not per-CTA dynamic issue or DRAM bytes. A CTA chooses one branch, keeps full-K P through both chronologicalK16 updates perBK32 and publishes complete P before the separate U combine. Ncu was not retried after the existing session lock failure.

Default GPU validation passes with2034 CASE/803 PASS, including four new independent integer A7/B7/P fixtures and36 random same-original-input paired H11↔H9/H12↔H10 complete-seven-P/final-D bit comparisons, plus retained Mega7 final-D reference. Full default validation has88 captures/264 freshAB replays; the matched additions contribute8/24 and cover beta-zero/nullC0 and active C0=D with refreshed original A/B and reset FP64 reference outside capture on the same stream. Runtime/metadata negatives, padded/tail/transpose/budget fallbacks, AB16 rebindings and scalar D/C0+4 are checked. The independent M2 stage reference remains separate from M1 bitwise comparisons.

Strict numerics records1290 unique gates/430 distributions:774 pass,516 fail (258uniform+258range),NF0,exit2. The unchanged first1230 gates and new60 matched checks are separate groups; new60 comprises48 native and12 required HighHalf Dense checks. Exact structural and paired-bitwise success does not replace this original-input numerical gate. CUDA12.8 Compute Sanitizer2025.1.0.0 independently runs H11/H12 memcheck/racecheck/synccheck, all six exit0 with zero leaks/errors/hazards/warnings. Each H11 run has44CASE/37PASS; each H12 has46CASE/37PASS; each includes2 direct input/product fixtures,18 full-P/final-D pairs and4 captures/12 freshAB replays. All runs are serial and completed before closure.

Final source/binary/raw/config/test/compiler evidence is [results/small_matched_profile_iter29.json](results/small_matched_profile_iter29.json). Benchmark SHA256 `66bcc473b87bc5d51f46af9b9b6012ae7dace43c32d2f0c594bbf31e30206d5e`; tests `fae22ec4c77586bbbc9d34fa1820903e60ebaef08756b75dbf36eec442c31671`. Source remains the actual measured `bdbcdd9`, not any later evidence-only HEAD. The128/256/512/1024 faster-than-cuBLAS objective remains unmet. These results do not establish a hardware floor, exhausted directions or completion of continuous optimization.

### Reassessment before iteration30 — active1024 megakernel goal

The new explicit user goal is to keep optimizing the1024 megakernel until it exceeds independently tuned cuBLAS. The preceding goal turn made progress: it integrated and verified matched input-placement controls, produced24 full1024 rows with72 fresh checks and immutable evidence, and closed checkpoint `62798a9`. It did not achieve the goal. Packed Lt1 or a Dense fallback is not a substitute for the requested single-launch megakernel.

Iterations27–29 produced no >=3% improvement over the overall historical own-SW envelope. AKO4ALL reassessment therefore precedes another formal implementation. The current same-source29 ordinary-launch fastest SW is Lt1 at19.3750/19.7350us (BF16/FP16), versus its independently tuned Dense12.2298/9.9302us; all original numerical gates fail. The controlled32x64 leaf experiment lowers complete SW latency38.2541/38.5088→34.1165/34.4358us when transforms move to once-per-call packing. This validates input placement as a useful axis but does not explain dynamic stall percentages or close the leaf-throughput gap. Source27 cooperative Mega27 is still a true single-launch option, while the current packed controls require multiple launches.

The existing ncu lock failure is not re-probed. Fresh Nsight Systems2023.4.4 CUDA-profiler capture-range diagnostics instead ran the unchanged source29 binary on allocation107488/g08, verified H100 UUID `GPU-78def847-b6c1-9dd4-ff30-269a9ae1a73f`,132SM,CUDA runtime12040. Both captures contain exactly one complete native megakernel and zero workspace: Mega0 grid8x16/128threads,150 registers/thread,dynamic shared reported0.043MB; Mega27 grid16x16/128threads,128 registers/thread,static shared0.014MB. The captured single diagnostic durations130911ns/30591ns are instrumented observations, not full benchmark medians, speedup scores, or stall attribution. Raw reports and trace CSVs are `results/reassessment_mega{0,27}_1024_20261014*` and retain the unchanged benchmark binary SHA above.

Source inspection confirms the old M1 still has a single128-thread warpgroup doing scalar input transforms, seven WGMMA products, immediate wait and stage reuse; no dedicated producer pipeline exists. Prior axes include small owner/product/cluster routes, raw-lookahead/cp.async owners, cooperative SM80 sharing, and two-/three-launch input-placement controls. A distinct next axis is a single-launch256-thread M1 with a128-thread producer warpgroup and128-thread consumer warpgroup, two final A7/B7 operand banks, vectorized original-quadrant loads/stores, and pinned `PipelineAsync<2>` full/empty handoff. Compare BM64 BN16/32 BK32:256/128 ownerCTAs at1024,56/112 theoretical P registers/consumer,70/84KiB final operand banks. BN16 adds parallelism but also increases logical original requests80MiB versus48MiB and FP32 input-DAG work41,943,040 versus25,165,824; none of these static counts predicts a win. Producer64/consumer160 dynamic-register settings remain a hypothesis until compilation and role-specific safety checks.

Primary-source rereading supports the implementation mechanism, not an H100 FMM speedup promise: [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.4.0/hopper-tuning-guide/index.html), [PTX8.4 WGMMA/fence/register protocols](https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html), [pinned CUTLASS3.5.1 PipelineAsync](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/pipeline/sm90_pipeline.hpp), and [GPU Strassen fusion research](https://arxiv.org/abs/1808.07984). TMA aliasing raw A21/B12 into future transformed slots, further stages/outstanding groups, and larger cooperativeSM80 tiles are retained independent later axes; they are not claimed implemented in this first pipeline experiment. No evidence establishes a performance floor or exhausted viable directions.

The original one-cast BF16/FP16 strategy and fixed gate remain in force. Read-only reevaluation of the existing exact-leaf witness matches its frozen JSON; lifting each2x2 coefficient matrix withI512 gives legal1024 metadata and the same Frobenius cast-error witness (a mathematical lifting, not a new GPU check). This explains why faster rounded-leaf execution alone cannot certify the full declared numerical range. An asynchronous user preference question about adding a separate higher-precision/compensation strategy is pending; until an answer, no such strategy is substituted. Execution-performance experiments continue, and numerical failures remain visible. This limitation does not establish a kernel-time floor.

The next iteration must preserve single-launch ownership/full-K sevenP/fixedU/alpha-beta/currentbinding/Graph-fresh-input semantics, the original FP64 reference and fixed gates, the old25CUDA files and frozen source23–29 evidence. Same-source full1024 comparison of newMega28/29, oldMega0/Mega27 and Lt1 controls in both dtypes/modes is required; own-Mega latency and the overall SW envelope are reported separately. Full20warm/100calls/5groups/3fresh, original checks and tuned same-row Dense must precede any winner or completion claim.

### Iter 30 — Single-launch producer-consumer WGMMA pipeline

- **Hypothesis:** Overlap FP32-RN input transformation and complete seven-P WGMMA through two final shared-memory banks without global S/T/P materialization.
- **Changes:** Add native Mega28/29 BM64 BN16/32 BK32 128-producer/128-consumer PipelineAsync2, safe compiled64/96-or160 register quotas, common current-binding Plan fallback and focused integer/raw-U/host-epilogue/Graph tests; preserve25oldCUDA sources, original numerical gate and AutoDense.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.358867 ms; mean speedup=0.4094x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.005854 | 0.006222 | 0.9409 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.005944 | 0.006392 | 0.9299 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.005952 | 0.008853 | 0.6723 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.005840 | 0.008949 | 0.6526 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.006322 | 0.013643 | 0.4634 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.006390 | 0.013869 | 0.4608 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.013205 | 0.031597 | 0.4179 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.010294 | 0.031763 | 0.3241 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.032037 | 0.216226 | 0.1482 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.032958 | 0.221128 | 0.1490 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.087035 | 0.820659 | 0.1061 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.089110 | 0.811558 | 0.1098 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.195374 | 2.080285 | 0.0939 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.205389 | 2.097142 | 0.0979 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.012040 | 0.022675 | 0.5310 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.011894 | 0.023114 | 0.5146 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.008438 | 0.022570 | 0.3739 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_wgmma_pipeline_64x16 v28 | launch | 0.008816 | 0.022966 | 0.3839 | 0.00075151 | 0 |

- **Analysis:** Predeclared nine-shape two-dtype launch screen20warm20calls5groups1fresh completed18finite numerical-failing rows exit2 after defaultGPU2080CASE859PASS and selected29same-input18raw-U/host-epilogue groups PASS. Native source and binary identities were frozen before measurement. This diagnostic alone establishes no certification or own-latency envelope gain; mandatory same-binary control and20row1024 full comparison follow.
- **Next:** Measure Mega29 same-binary quick control and both new pipelines, Mega0/Mega27/Lt1 in launch/Graph20warm100calls5groups3fresh; finish original strict60scope, sixCUDA12.8sanitizers, compiled resource and independent evidence audit, then report actual endpoint latency and numerical failure.
- **Archive:** trajectory/20261014_005808_iter-30/

#### Iter30 full1024 evidence closure

Same frozen source checkpoint `4a655fc7ed4c30497d262c7dd2a82137b2833989`, seven source hashes and bench/test SHA `ccad7e48…`/`140a9518…` cover both native pipelines, retainedMega0/Mega27 and multi-launchLt1 in launch/Graph. Predeclared20warm100calls5groups3fresh completed20rows/60fresh/200TIME; all original numerical gates fail/NF0, each benchmark exits2. Full raw/statistics/contracts/resource/report binding is `results/small_pipeline_profile_iter30.json`, with expanded measurements in `results/small_pipeline_measurements_iter30.json` and `results/1024_MEGAKERNEL_PIPELINE_REPORT.md`.

| Mode | dtype | RetainedMega27 own µs | NewMega29 own µs | Own latency decrease | Same-rowDense µs | Dense/SW | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| launch | bf16 | 31.0838 | 25.1098 | 19.219% | 12.1194 | 0.482655 | FAIL finite |
| launch | fp16 | 30.1795 | 25.4029 | 15.827% | 9.8550 | 0.387950 | FAIL finite |
| graph | bf16 | 30.6342 | 24.8368 | 18.925% | 11.3440 | 0.456742 | FAIL finite |
| graph | fp16 | 30.2394 | 25.0938 | 17.016% | 9.4438 | 0.376342 | FAIL finite |

Mega29 improves the retained same-binary trueMegakernel comparison, but has not exceeded cuBLAS. Mega28 offers no improvement overMega27. All measuredSW route winner remains multi-launchLt1 at19.2739/19.6486µs ordinary and16.8362/17.1894µs Graph; it is a control and cannot replace the single-launch active goal. The historical source23 allSW envelope is19.6432/19.3498µs Lt1 at1024; the new pipeline does not improve that historical envelope. Ordinary/Graph and ownSW/Dense ratios remain separate; no floor or direction-exhaustion claim.

Default2080CASE859PASS, selected29 24CASE38PASS, CPU3DAG+6stage PASS; independent tests audit preserves old2034CASE/803PASS domains and34old function bodies. New29 eighteen same-input raw-U/final-D fixtures pass against matchingN32oldMega0 rawU plus independenthostRNmultiply/floatstd::fma epilogue. An early activebeta comparison failed because oldMega0 contracts beta*C0 first; olddiagnostic and SASS proofs are retained. Device specifiedalphaRNbetaFMA and originalFP64gate were not changed. No direct seven-P exportedbitwise claim for this test or N16/N32bitwise claim. NewGraph12configurations36freshAB replay eachcaptureonekernelnode PASS.

Newstrict60 scope retains original floors/patterns:36PASS24FAIL/NF0,48native12Dense-rangefallback; old1290gates and25oldCUDA TUs/source23–29 evidence remain bytepreserved, not a repeated1350suite. SixCUDA12.8sanitizers for28/29 eachmemcheck/racecheck/synccheck exit0 withzeroerrors/hazards/warnings/leaks. Static gate beforelaunch proves BN16REG96/pool24576≥role20480, producer64/consumer96 withmaxR58/89; BN32REG160/pool40960≥28672, quotas64/160 maxR54/145 includingimplicitwidths andwaitstubs. STACK/LOCAL/spill0; formal dynamicshared71808/86144B versusELFstaticSHARED1024B separately recorded. Four linkedinstructionstreams equal finalobject; resource conditions forBN16twoCTA are not measuredoccupancy. Ncu not retried.

Original BF16/FP16 precision/numerical contracts and AutoDense remain unchanged. Continue optimizing the actual1024 single-launch kernel; no next implementation started in this lease before evidence closure.

### Iter 31 — Single-launch TMA raw-alias two-bank WGMMA

- **Hypothesis:** Eight direct TMA copies and raw shared-slot aliases replace repeated producer SIMT global loads and original-slot re-stores while preserving whole seven-product one-level SW.
- **Changes:** Add native Mega30/31 BM64 BN16/32 BK32 producer128 consumer128 PipelineAsync2; two current-pointer tensor-map encodes per complete ordinary invocation; original precision and fixed FP64 gate, truthful packed7/Dense fallback and AutoDense retained. Actual pinned-layout assertion domain repaired without changing executable host/device paths; full new object/linked quota and descriptor lowering independently reviewed before GPU.
- **Bench:** Compiled=True; frozen numerical gate=FAIL/unchecked. No certification is inferred from runtime gains.
- **Scope:** 18 rows; arithmetic mean of row median latency=0.292840 ms; mean speedup=0.3970x. Compare individual identical shape/dtype/mode rows below.

| M,N,K | dtype | route | mode | dense ms | solution ms | speedup | SW eF | numerical pass |
|---|---|---|---|---:|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.005621 | 0.006515 | 0.8627 | 0.00605216 | 0 |
| 128,128,128 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.005574 | 0.006728 | 0.8285 | 0.000744325 | 0 |
| 256,256,256 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.005677 | 0.009056 | 0.6269 | 0.00597914 | 0 |
| 256,256,256 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.005696 | 0.009256 | 0.6154 | 0.000737165 | 0 |
| 512,512,512 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.006349 | 0.013518 | 0.4696 | 0.00598238 | 0 |
| 512,512,512 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.006394 | 0.013870 | 0.4610 | 0.000753207 | 0 |
| 1024,1024,1024 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.013240 | 0.027469 | 0.4820 | 0.00600975 | 0 |
| 1024,1024,1024 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.010250 | 0.027630 | 0.3710 | 0.000751819 | 0 |
| 2048,2048,2048 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.031845 | 0.197074 | 0.1616 | 0.00602393 | 0 |
| 2048,2048,2048 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.032891 | 0.200723 | 0.1639 | 0.000752202 | 0 |
| 3072,3072,3072 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.086694 | 0.653165 | 0.1327 | 0.00601824 | 0 |
| 3072,3072,3072 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.088802 | 0.651856 | 0.1362 | 0.000752812 | 0 |
| 4096,4096,4096 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.195477 | 1.644518 | 0.1189 | 0.00602272 | 0 |
| 4096,4096,4096 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.205923 | 1.713634 | 0.1202 | 0.000753133 | 0 |
| 1024,512,1024 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.011522 | 0.023886 | 0.4823 | 0.00599042 | 0 |
| 1024,512,1024 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.009880 | 0.024395 | 0.4050 | 0.000749333 | 0 |
| 512,1024,1024 | bf16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.008323 | 0.023563 | 0.3532 | 0.00601885 | 0 |
| 512,1024,1024 | fp16 | megakernel_wgmma_tma_alias_64x16 v30 | launch | 0.008632 | 0.024262 | 0.3558 | 0.00075151 | 0 |

- **Analysis:** Default GPU2126CASE933PASS, selected30/31 both38PASS, CPU DAG3 stage6 PASS, independent initial CTApool/direction/role CFG including implicit widths and descriptor/shared alignment PASS. This diagnostic quick uses18rows20warm20calls5groups1fresh and records all numerical failures; matched full20row1024 comparisons and host batchwall are required separately irrespective screen signal. No numerical certification or goal completion inferred from execution timing.
- **Next:** Measure same frozen source/binary new30/31 and retained29/27/Lt1 at1024 bothdtype ordinary/Graph20warm100calls5groups3fresh, then unchanged strict60 and six serial CUDA12.8 sanitizers, independent static/data final audit and evidence closure.
- **Archive:** trajectory/20261014_013912_iter-31/

#### Iteration31 full1024 follow-up

Same frozen source `20363e90a7a936e80f471fede0e16b96855ca11f` / bench SHA `9e3b6a7245217aa4f728622961e9b45640346fbf4945ed76019eef472c200785` measured five routes (newTMA30/31, retainedMega29/27, multi-launchLt1) × BF16/FP16 × ordinary/Graph,20warm100calls5groups3fresh. Raw20rows/200TIME/60freshCHECK are retained, all originalFP64gate FAIL/NF0; no Auto promotion or goal completion.

| Mode | dtype | Retained29 event µs | New31 event µs | New31 own latency increase | New31 Dense/SW | Retained29/new31 batchwall µs/call |
|---|---|---:|---:|---:|---:|---:|
| ordinary | bf16 | 25.239041 | 25.702400 | 1.836% | 0.476880 | 25.31608/25.77972 |
| ordinary | fp16 | 25.427520 | 26.413441 | 3.877% | 0.372029 | 25.50958/26.49597 |
| Graph | bf16 | 24.776001 | 24.929600 | 0.620% | 0.452782 | 24.85355/25.00263 |
| Graph | fp16 | 25.054400 | 25.605760 | 2.201% | 0.365843 | 25.13400/25.67863 |

New31 beats new30, but retained29 remains faster in all four combinations. Multi-launchLt1 remains the observed allSW control winner (ordinary19.373120/19.501120µs; Graph17.002881/17.225280µs), distinct from trueMegakernel target. Ordinary includes two actual currentpointer tensor-map encodes per complete hostloop invocation then one kernel; batchwall/call includes submission and batchsync, not independent synchronized singlecall wall or isolated encode time. Graph captures fixed descriptor bindings and freshly recomputes all input transforms each replay; no host encode on replay.

DefaultGPU2126CASE933PASS, selected30/31 22/24CASE38PASS, CPU DAG3+stage6 PASS, new36 sameatom rawU/host specifiedepilogue groups and40currentbindingchecks PASS, new12Graph36fresh one-node replays PASS. New strict60=36PASS24FAIL/NF0 (48native12Dense rangefallback); six targeted CUDA12.8SAN exit0,0errors/0racehazards/warnings, mem2 leak0. Structure/bitwise checks do not replace original numerical certification. Final report/profile/raw evidence and independent compiled/data audits are archived; original26 CUDA/source23–30 bytes retained. TMA31 performance status is regression against retained29, not an improvement. No floor, exhausted-direction, isolated traffic or encode causal claim.

### Iter 32 — Overlap two committed WGMMA groups with vector producer

- **Hypothesis:** Retain identical vector producer/tile/whole-sevenP; consumer wait1 retires the old bank while latest group remains in flight, reducing per-stage forced wait0.
- **Changes:** Native Mega32/33 BM64 BN16/32 BK32 PipelineAsync2 with separate read/retire states and final wait0 before genericP/U. Original singlecast RN S/T, complete7P/fixedU, alphaRN betaFMA, public binding/overlap guards, packed7/Dense fallback and AutoDense retained. All27 prior CUDA files unchanged. Independent new compiled quota/inflight-P gate passed before firstlaunch.
- **Bench:** Compiled=True; quick18rows20warm20calls5groups1fresh; original numerical failures retained; no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.920000 | 7.444800 | 1.332474 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.848000 | 7.142400 | 1.378808 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.880000 | 9.448000 | 1.045724 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.945600 | 9.593600 | 1.036691 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 10.232000 | 15.291201 | 0.669143 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.982400 | 15.288000 | 0.652957 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 17.435201 | 33.344001 | 0.522889 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 10.657600 | 33.230400 | 0.320718 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 31.747201 | 233.159995 | 0.136161 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 33.081600 | 240.659189 | 0.137462 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 87.896001 | 897.497559 | 0.097935 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 89.897603 | 880.039978 | 0.102152 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 195.787203 | 2204.065514 | 0.088830 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 205.377603 | 2227.649689 | 0.092195 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.512000 | 25.703999 | 0.370059 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 9.553600 | 25.889599 | 0.369013 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_wait1_64x16 v32 | launch | 8.272000 | 25.681600 | 0.322098 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_wait1_64x16 v32 | launch | 8.585600 | 25.705600 | 0.333997 | FAIL |

- **Analysis:** Quick signal only. Rank new own latency against frozen29 samebinary, not Dense denominator. Default/targeted structural checks remain separate from original numerical gate.
- **Next:** Mandatory full1024 20rows new32/33 retained29/TMA31/Lt1 bothdtype ordinary/Graph20warm100calls5groups3fresh, unchanged newstrict60 and six serial CUDA12.8 sanitizers; independent static/data audit and full evidence closure.
- **Archive:** trajectory/20261014_020836_iter-32/

#### Iteration32 full1024 evidence closure

Same frozen source `f396662f498af7998b2401da49506184b96904ae` / bench SHA `8fe268ef4a42f51bcea88478d3451c69bc963b245a267db288ef24d5de1f3dab` measured20rows200TIME60freshCHECK with newwait1 Mega32/33, retained29/TMA31 and multi-launchLt1, bothdtypes ordinary/Graph20warm100calls5groups3fresh. All60originalFP64gateFAIL/NF0; noAutochange orgoalcompletion.

| Mode | dtype | Retained29 event µs | Bestnew33 event µs | New33 ownlatency increase | New33 Dense/SW | Retained29/new33 batchwall µs/call |
|---|---|---:|---:|---:|---:|---:|
| launch | bf16 | 25.134721 | 29.158080 | 16.007% | 0.420703 | 25.21137/29.25172 |
| launch | fp16 | 25.401919 | 29.407041 | 15.767% | 0.337084 | 25.47930/29.48424 |
| graph | bf16 | 24.854400 | 28.604801 | 15.089% | 0.396633 | 24.93586/28.68783 |
| graph | fp16 | 25.129600 | 28.983040 | 15.334% | 0.325929 | 25.20803/29.07212 |

New33 beatsnew32 butregressesretained29 inallfour combos. New33vs29 sameBM64BN32BK32 vectorproducer isolates consumerschedule;32vs29 differentBN cannotaloneattribute schedulecost. Lt1 remainsfastestobservedmulti-launchcontrol andcannotcomplete trueMegakernelgoal. Ordinarywait1/retained29 nohostencode; retainedTMA31 two currentpointerencodeinsidehostloop; Graphcapturedaddresses freshcontents eachreplay. Eventmayincludehostsubmissionidle; batchwall/call includes100submissions+batchsync, notindependentsinglecallwall.

Default2188CASE1075PASS, new62CASE142PASS; selected32/33 actual30/32CASE72PASS. New96sameatom rawU/host specifiedepiloguegroups,40currentABchecks,20Graph60freshnode1 replays allPASS, coveringstage1/2/3/4/16/17/32/33 andtail/phasewrap. Strictnew60=36PASS24FAILNF0 (48native12Denserangefallback); six serialCUDA12.8SANexit0zeroerrors/racehazardswarnings/memleaks. Originalsinglecastprecision andgates retained. Historicalsource29full1290 plus30/31actualnew60each preserved underindividualidentity; source30full1350/source31full1410/currentfull1470 areunexecuted code-scopecounts. Newownedpool/rolefullCFG/implicitwidth/accinit/zero genericPaccesswhileinflight/linkedobject exactinstruction gatePASS beforelaunch; no31/30safetyinheritance. Compiled1168/1504staticinstructions andquota96/160 do notmeasuredoccupancy/traffic. Nohardwarefloor orcausalstallproof claimed. Finalprofile/report/independentdataandstaticaudit closurefollows; nextaxis notstartedunderthislease.

### Iter 33 — Add third operand bank to vector wait1 pipeline

- **Hypothesis:** A third producer-filled bank offers additional producer slack while two committed WGMMA groups remain unretired; prior timing did not isolate a two-bank stall mechanism.
- **Changes:** Native Mega34/35 BM64 BN16/32 BK32 PipelineAsync3, full128/empty128, separate read/retire states and finalwait0 before genericP/U. Only bankcount changed; no TMA/release1/twoConsumer/Psplit. OriginalsinglecastRN S/T, complete7P/fixedU, alphaRNbetaFMA, currentbinding/overlap, packed7/Dense fallback andAutoDense preserved. Old28CUDA bytefrozen. Independentnewcompiled quota/modulo3/tail3/inflight-P/object-linked gate beforefirstlaunch.
- **Bench:** Compiled=True; quick18rows20warm20calls5groups1fresh; original numericalFAIL retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 5.939200 | 6.337600 | 0.937137 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 5.924800 | 6.508800 | 0.910275 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 5.977600 | 8.926400 | 0.669654 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 5.897600 | 9.043200 | 0.652158 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 6.566400 | 13.480000 | 0.487122 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 6.598400 | 13.769600 | 0.479201 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 13.257600 | 31.817600 | 0.416675 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 10.249600 | 32.275200 | 0.317569 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 31.756800 | 219.774389 | 0.144497 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 32.784000 | 225.364804 | 0.145471 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 87.238401 | 850.044823 | 0.102628 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 89.286399 | 838.216019 | 0.106520 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 195.673597 | 2096.174431 | 0.093348 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 206.547189 | 2123.124886 | 0.097285 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 11.518400 | 22.300801 | 0.516502 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 11.972800 | 22.920001 | 0.522373 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 8.419200 | 22.156800 | 0.379983 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_wait1_three_stage_64x16 v34 | launch | 8.769600 | 22.728001 | 0.385850 | FAIL |

- **Analysis:** Quicksignal only. Compare ownlatency againstsamebinary retained29 andsamewidthtwo-bank33, notslowerDense denominator. Structural/rawU/hostepilogue/Graph checks separateoriginalFP64gate.
- **Next:** Mandatory full1024 new34/35+29+two-bank33+Lt1 bothdtype ordinary/Graph20warm100calls5groups3fresh, strictnew60 andsixserialCUDA12.8SAN, independentstatic/datafinal audit andcompleteevidenceclosure.
- **Archive:** trajectory/20261014_023614_iter-33/

#### Iter33 full1024 and validation closure

Same measured source `96938eb558dd031d37ffc6f7f3ff74f57736de9c`, one frozen bench and original precision/FP64 gates; new34/35, retained29, two-bank33 and multi-launchLt1, both dtypes/ordinary+Graph. All20rows/200TIME/60freshCHECK complete; numerical60FAIL/NF0, no certification or Auto switch. Times are median5group means of100 completecalls,20warm,3fresh,32 independentcuBLASLt heuristics8GiB. Batchhostwall includes submission+batchsync divided100, not independently measured singlecallwall.

| Mode | dtype | 3bank35 µs | 2bank33 µs | own decrease vs33 | retained29 µs | own change vs29 | same-row Dense/SW35 |
|---|---|---:|---:|---:|---:|---:|---:|
| launch | bf16 | 25.407679 | 29.096961 | 12.679269% | 25.163200 | +0.971574% | 0.483596 |
| launch | fp16 | 25.605760 | 29.248321 | 12.453916% | 25.537920 | +0.265643% | 0.384863 |
| graph | bf16 | 24.951680 | 28.640001 | 12.878216% | 24.869440 | +0.330687% | 0.454959 |
| graph | fp16 | 25.332479 | 28.880639 | 12.285599% | 25.033600 | +1.193912% | 0.371785 |

A third bank recovers12.29–12.88% of35latency versus sameBN32 two-bank33, while observedmedians are0.27–1.19% slower thanretained29. This shows this implementation benefits from the thirdbank; it does not isolate an individual hardware stall mechanism or prove a performance floor. AllSW fastest is Lt1 multiplelaunchcontrol, not trueMegakernelgoal.

Actual GPU default2262CASE1269PASS, selected34=36CASE98PASS and35=38CASE98PASS, all exit0; new74CASE194PASS,132sameatom rawU+specifiedhostepilogue,40currentABchecks,20outputrebindgroups60runs,new36Graph108fresh(all168/504), singlekernel node and stages1/2/3/4/6/7/16/17/18/33/34 verified. Strictnew60=36PASS24FAILNF0exit2,48native12Densefallback; source29actualfull1290 andsource30/31/32actualnew60 retained by separateidentity; source30full1350/31full1410/32full1470/currentfull1530 remain unexecuted fullscopecounts. SixserialCUDA12.8SANexit0, memcheckerrors/leakedbytes0,racehazards/errors/warnings0,synccheckerrors0; leaktestonlymemcheck.

Freshfourinstancecompiledgate independentlyPASS, completeobject/bench/test address+predicate+instructionstreams5424 (raw objectheaderflags differ), REG96/160, prodhighestR58/54 conshighestR89/145 includingimplicitwidth, initialownedCTA24576/40960 meetsrolepool20480/28672, directionslegal, allaccinitialized andgenericPaccess0 untiltailwait0. Dynamic107648/129152B+static1024B align128, stack/local/spill0; actualmod3phase/six8Bbarriers/3producertailacquires checked. CPUabstractmodelstdout+realexit0 archived14279states27569transitions395523phasecrosschecks; notGPUhardwarecertificate. All28oldCUDA/source23–32 evidence bytepreserved. ncu not retried after priorlockfailure. Fullrowstats and allrawbinds: results/small_three_stage_measurements_iter33.json, results/small_three_stage_profile_iter33.json, results/1024_MEGAKERNEL_THREE_STAGE_REPORT.md. Independentfinalaudits freeze separately. Originalgoal remainsactive andAutoDense.

### Iter 34 — Dual consumer P3+4 experiment with actual compiled safety gates

- **Hypothesis:** Split whole-K P1–3/P4–7 across two independent consumer warp groups while one vector producer constructs S/T once per tile and shares the final two operand banks.
- **Changes:** Both variants use BM64 BN32 BK32,384 threads, full128/empty256, whole seven P, original fixed U/alphaRNbetaFMA, no global S/T/P workspace. Source/budget revisions and blocked routes are separately frozen; only independently gated kernels may execute. Consumer-only q scope uses a nonaligned drain CTA barrier with count384, followed by complete P publication and shared U. Old29CUDA preserved. Original numerical gate and AutoDense unchanged.
- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 5.779200 | 6.398400 | 0.903226 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 5.812800 | 6.528000 | 0.890441 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 6.072000 | 9.025600 | 0.672753 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 5.884800 | 9.096000 | 0.646966 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 6.582400 | 14.284800 | 0.460798 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 6.350400 | 14.270400 | 0.445005 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 13.177601 | 23.748800 | 0.554874 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 10.254400 | 23.667200 | 0.433275 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 32.011199 | 197.489595 | 0.162091 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 33.105600 | 202.563190 | 0.163433 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 87.111998 | 752.070379 | 0.115830 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 88.896000 | 741.963196 | 0.119812 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 195.400000 | 1918.872070 | 0.101831 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 206.367993 | 1935.454369 | 0.106625 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 11.627200 | 24.097601 | 0.482504 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 11.700800 | 23.119999 | 0.506090 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_64x32 v36 | launch | 8.198400 | 23.032001 | 0.355957 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_64x32 v36 | launch | 8.478400 | 23.123200 | 0.366662 | FAIL |

- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.
- **Next:** Mandatory full1024 measured safe routes plus retained29/35/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.
- **Archive:** trajectory/20261014_032834_iter-34/

### Iter 35 — BK64 P3+4 geometry with fresh compiled safety gate

- **Hypothesis:** Increase BK32 to64 while preserving the successful36 P3+4 dual-consumer scope; halve K-stage count at1024, with four ascendingK16 steps perP perstage, alloriginaltransforms and epilogue insideeach completekernel.
- **Changes:** New38 uses BM64 BN32 BK64,384threads/producer64/WG96/WG96,full128/empty256/two banks. OnlyBK/A-vectorindex/parentKguardgeometry changes plus distinctrename; whole7P/fixedU/alphaRNbetaFMA/consumer-onlyq/nonaligneddrain6/zeroGWS unchanged. NativeparentK%128; partialKfallback truthful. Old31CUDA preserved, registered37 disabled. Originalnumericalgate/AutoDense unchanged; onlynewfreshcompiledgatedkernels execute.
- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 5.572800 | 6.244800 | 0.892390 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 5.636800 | 6.406400 | 0.879870 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 5.686400 | 8.860800 | 0.641748 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 5.684800 | 8.926400 | 0.636853 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 6.296000 | 13.880000 | 0.453602 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 6.315200 | 13.944000 | 0.452897 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 13.424000 | 24.048001 | 0.558217 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 10.320000 | 24.291199 | 0.424845 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 31.654400 | 196.660805 | 0.160959 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 32.644799 | 202.663994 | 0.161078 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 86.609602 | 719.028807 | 0.120454 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 89.007998 | 714.793587 | 0.124523 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 194.755197 | 1864.990425 | 0.104427 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 205.999994 | 1908.491135 | 0.107939 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 11.585600 | 24.040000 | 0.481930 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 11.489600 | 24.320000 | 0.472434 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 8.188800 | 23.176000 | 0.353331 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_bk64_64x32 v38 | launch | 8.512000 | 23.424000 | 0.363388 | FAIL |

- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.
- **Next:** Mandatoryfull1024 measured38 plus retainedbest36/29/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.
- **Archive:** trajectory/20261014_040812_iter-35/

### Iter 36 — RS-A producer rawA4/sharedB7 and consumer register A

- **Hypothesis:** Place A transformation in consumers using RS-A while preserving whole-K seven P and fixed U; each stage builds distinct packed A words before issue and retains every A operand until wait0.
- **Changes:** New39 uses BM64 BN32 BK64,384threads/producer64/WG192/WG192,full128/empty256/two banks; producer rawA4 and finalB7, consumer complete-stage packed A48/64 plus P48/64 words. Actual initial quota/resource/full pendingA+P gate precedes any GPU launch. Original onecast/fixedU/alphaRNbetaFMA/nonaligned drain6/zeroGWS unchanged. ParentK%128 and truthful fallback. Old32CUDA preserved, registered37 disabled. Originalnumericalgate/AutoDense unchanged; onlynewfreshcompiledgatedkernels execute.
- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 5.920000 | 6.808000 | 0.869565 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 5.828800 | 6.982400 | 0.834785 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 5.998400 | 9.099200 | 0.659223 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 6.572800 | 9.203200 | 0.714186 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 6.476800 | 13.708800 | 0.472456 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 6.384000 | 13.832000 | 0.461538 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 13.278399 | 23.391999 | 0.567647 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 10.214400 | 23.212799 | 0.440033 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 31.972799 | 187.769604 | 0.170277 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 33.287999 | 184.072006 | 0.180842 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 87.055999 | 685.721588 | 0.126955 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 89.558399 | 681.411219 | 0.131431 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 195.553601 | 1797.873688 | 0.108769 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 206.177592 | 1798.646355 | 0.114629 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 9.500800 | 23.337600 | 0.407103 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 9.632000 | 23.393600 | 0.411737 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 8.334400 | 23.310401 | 0.357540 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_64x32 v39 | launch | 8.681600 | 23.321600 | 0.372256 | FAIL |

- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.
- **Next:** Mandatoryfull1024 measured39 plus retainedbest36/sameBK64SS38/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.
- **Archive:** trajectory/20261014_045025_iter-36/

### Iter 37 — RS-A producer rawA4/sharedB7 and consumer register A

- **Hypothesis:** Use BK32 RS-A with the verified numeric ARLayout adapter while preserving whole-K seven P and fixed U; each stage builds distinct packed A words before issue and retains every A operand until wait0.
- **Changes:** New40 uses BM64 BN32 BK32,384threads/producer64/WG192/WG192,full128/empty256/two banks; producer rawA4 and finalB7, consumer complete-stage packed A24/32 plus P48/64 words. Actual initial quota/resource/full pendingA+P gate precedes any GPU launch. Original onecast/fixedU/alphaRNbetaFMA/nonaligned drain6/zeroGWS unchanged. ParentK%64 and truthful fallback. Old33CUDA preserved, registered37 disabled. Originalnumericalgate/AutoDense unchanged; onlynewfreshcompiledgatedkernels execute.
- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 5.656000 | 6.790400 | 0.832941 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 5.681600 | 6.889600 | 0.824663 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 5.689600 | 9.691200 | 0.587089 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 5.897600 | 9.531200 | 0.618768 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 6.328000 | 15.372799 | 0.411636 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 6.321600 | 14.916800 | 0.423791 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 13.123199 | 25.702399 | 0.510583 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 10.270400 | 24.417600 | 0.420615 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 32.032001 | 209.279990 | 0.153058 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 32.991999 | 205.879998 | 0.160249 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 86.857599 | 779.529619 | 0.111423 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 88.988799 | 745.988798 | 0.119290 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 194.836795 | 1935.716820 | 0.100654 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 206.308794 | 1952.607918 | 0.105658 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 11.478400 | 26.462400 | 0.433763 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 11.660800 | 24.105600 | 0.483738 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 8.177600 | 25.200000 | 0.324508 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_bk32_64x32 v40 | launch | 8.481600 | 24.150400 | 0.351199 | FAIL |

- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.
- **Next:** Mandatoryfull1024 measured40 plus retainedRSBK64best39/matchedBK32SS36/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.
- **Archive:** trajectory/20261014_052434_iter-37/

### Iter 38 — Hybrid S1/S2 producer and complete-stage register A

- **Hypothesis:** Use hybrid S1/S2 producer construction with original RN onecast and BK32 RS-A while preserving whole-K seven P and fixed U; each stage builds distinct packed A words before issue and retains every A operand until wait0.
- **Changes:** New41 uses BM64 BN32 BK32,384threads/producer64/WG192/WG192,full128/empty256/two banks; producer rawA4, final roundedS1/FP32S2 planes and finalB7, consumer complete-stage packed A24/32 plus P48/64 words. Actual initial quota/resource/full pendingA+P gate precedes any GPU launch. Original onecast/fixedU/alphaRNbetaFMA/nonaligned drain6/zeroGWS unchanged. ParentK%64 and truthful fallback. Old34CUDA preserved, registered37 disabled. Originalnumericalgate/AutoDense unchanged; onlynewfreshcompiledgatedkernels execute.
- **Bench:** quick20warm20calls5groups1fresh; original numericalFAIL is retained, no certification.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 5.646400 | 7.137600 | 0.791078 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 5.614400 | 7.300800 | 0.769012 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 5.694400 | 10.408000 | 0.547118 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 5.766400 | 10.478400 | 0.550313 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 6.334400 | 16.816001 | 0.376689 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 6.372800 | 16.931200 | 0.376394 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 13.211200 | 29.441601 | 0.448726 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 10.214400 | 29.244801 | 0.349272 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 31.990400 | 238.459206 | 0.134155 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 32.734400 | 241.902399 | 0.135321 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 87.208003 | 884.284782 | 0.098620 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 89.279997 | 867.504025 | 0.102916 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 195.660806 | 2175.673676 | 0.089931 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 206.839991 | 2206.334305 | 0.093748 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 11.502400 | 29.921600 | 0.384418 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 11.649600 | 30.062401 | 0.387514 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 8.430400 | 29.912001 | 0.281840 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_hybrid_s_64x32 v41 | launch | 8.497600 | 28.716800 | 0.295910 | FAIL |

- **Analysis:** Own complete SW latency determines code benefit; same-row tuned Dense/SW reports the comparison. Numerical/structural/SAN evidence stays separate. Missing unsafe routes will have explicit compiled-safety exclusions rather than invented rows.
- **Next:** Mandatoryfull1024 measured41 plus retainedRSBK64best39/matchedrawBK32RS40/SS36/Lt1,100calls5groups3fresh; actual validation scope, independent implementation/data audits and source/raw/report/Git closure.
- **Archive:** trajectory/20261014_054901_iter-38/

### Iter 39 — Paired BN16 RS-A and shared/shared seven-product kernels

- **Hypothesis:** Compare BN16 RS-A42 and SS43 with complete ascending-K seven P and fixed U. Source quotas/min2 alone are hypotheses; fresh per-route compiled proof precedes launch.
- **Changes:** Two new BM64BN16BK32/two-bank384-thread TUs. RS42 quotas48/96/96 and SS43 quotas64/80/80; both min2. New literal tail permutation preserves unique P/U coordinates. Initial42 three scalar state words spilled; equivalent PipelineState derived from the unchanged int64 K counter removes persistent phase liveness. Initial failed source and scheduling error remain archived. Four final instances have actual REG80 zero stack/local/spill; all-role/pending-A-P/init/ABI/drain and object/bench/test identity are independently checked. OccupancyAPI resource prediction is separate from measured residency. Original ST onecast/DAG/alphaRNbetaFMA/floors/AutoDense remain unchanged.
- **Bench:** quick20warm20calls5groups1fresh; original numerical results retained.

| M,N,K | dtype | route | mode | Dense µs | complete SW µs | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 5.608000 | 5.851200 | 0.958436 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 5.606400 | 6.011200 | 0.932659 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 5.654400 | 8.555200 | 0.660931 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 5.614400 | 8.630400 | 0.650538 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 6.276800 | 13.432001 | 0.467302 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 6.312000 | 13.596800 | 0.464227 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 13.251200 | 29.948801 | 0.442462 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 10.292800 | 30.017599 | 0.342892 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 31.777599 | 214.979196 | 0.147817 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 32.916799 | 220.736003 | 0.149123 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 87.015998 | 847.192001 | 0.102711 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 89.499199 | 855.126381 | 0.104662 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 195.387197 | 2095.262337 | 0.093252 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 205.876803 | 2119.743919 | 0.097123 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 11.587200 | 22.468799 | 0.515702 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 11.670400 | 22.942400 | 0.508683 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 8.398400 | 22.428800 | 0.374447 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_ss_bn16_64x16 v43 | launch | 8.462400 | 22.164799 | 0.381795 | FAIL |

- **Analysis:** Compare own complete SW latency, then same-row tuned Dense/SW. Machine safety, structural fixtures and sanitizers do not certify the original FP64 numerical gate.
- **Next:** Mandatorysamebinary full24 for42/43/39/40/36/Lt1,100calls5groups3fresh irrespective quick3%; enabled strict30 per newroute and serialsanitizers, actualcoverage/report/profile/independentaudits.
- **Archive:** trajectory/20261014_062200_iter-39/

### Iter40 — BK64 RS-A single owning A microbatch

- **Hypothesis:** Reuse one complete-stage register A after each P commit/wait0, reducing owning A payload while preserving BM64BN32BK64/2banks/64-192-192/min1. More shared reads, RN S1 recomputation and waits can offset lifetime savings; no occupancy/performance improvement assumed.
- **Changes:** New44 only, each P four ascendingK16 then commit/wait0/postA-P fences; stage shared release only after all3/4P retired. Original producer/shared/tail/U, onecast/wholeK/alphaRNbetaFMA/floors/currentguards/AutoDense/blocked37 preserved. Fresh new44 all-role/initialpool/genericpending/init/ABI/linked gate precedes actual GPU.
- **Stall re-assessment:** Iter37–39 showed no >=3% own-latency gain; there is no measured physical-floor claim and prior ncu IPC failure is not retried. NVIDIA Hopper Tuning Guide describes TMA as reducing transfer instructions/register use; this iteration tests A lifetime instead. Strassen DAG fusion must be evaluated for the actual shape/GPU; published A100/H200 results are not H100 acceptance evidence. Sources: https://docs.nvidia.com/cuda/hopper-tuning-guide/ ; https://www.microsoft.com/en-us/research/publication/compiling-strassen-like-matrix-multiplication-algorithms-to-fast-cuda-kernels/ ; https://github.com/microsoft/subcuber/blob/main/README.md .
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 5.640000 | 7.192000 | 0.784205 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 5.595200 | 7.388800 | 0.757254 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 5.723200 | 9.603200 | 0.595968 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 5.652800 | 9.678400 | 0.584063 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 6.348800 | 14.705600 | 0.431727 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 6.432000 | 14.430401 | 0.445726 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 13.222399 | 24.257600 | 0.545083 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 10.422400 | 23.710400 | 0.439571 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 31.809601 | 193.569601 | 0.164332 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 33.025599 | 189.668798 | 0.174122 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 86.769599 | 710.915184 | 0.122053 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 89.716798 | 698.963213 | 0.128357 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 195.443201 | 1845.166397 | 0.105922 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 206.692791 | 1823.572731 | 0.113345 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 11.627200 | 24.211200 | 0.480241 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 11.612800 | 23.768000 | 0.488590 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 8.438400 | 24.171200 | 0.349110 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_rsa_microbatch_64x32 v44 | launch | 8.793600 | 23.641600 | 0.371955 | FAIL |

- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.
- **Next:** Mandatory samebinaryfull20 for44/39/40/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.
- **Archive:** trajectory/20261014_065139_iter-40/

#### Iter40 full closure evidence

Mandatory full20/200TIME/60CHECK completed;60 originalFAIL/NF0,10 innerexit2.44 ordinaryBF23.6076807976/FP23.2681608200us versus39 23.1145596504/23.1103992462us;Graph44 23.3548808098/23.1475210190 versus39 22.9222393036/22.9046392441us.44 failed to improve its samegeometryparent39; all20DenseCV<2% and allgroups kept. Default2690/2292 and selected56/135 exit0, strict30=18PASS12FAILNF0/exit2, SAN3zeroerror with leakonlymem0. RawU reference39/hostspecifiedepilogue/Graph/CPU abstractions are separate from originalFP64certification. No45/no precision relaxation/no promotion,AutoDense/goalactive. Single-A payload savings did not establish speedup or occupancy/stall cause. Exact current evidence is the microbatchreport/profile/measurement/coverage; future selectiveTMAA and role-clock prefixes are CPU-only and not current40 measurements.

### Iter41 — BK64 selective rawA4 TMA / B SIMT

- **Hypothesis:** Selective four raw A TMA transfers avoid SIMT A loads/stores; B transforms remain SIMT. Four8192-byte boxes and a separate raw32768-byte TX barrier, SW128/absolute1024 alignment; extra wait/encode costs can offset producer savings.
- **Changes:** New45 only, parent39 BM64BN32BK64/2banks/wholeP3+4/ascendingK16/onecast/fixedU/RNalphaFMA and original floors preserved. A-only lda<2^39 checked before map encode; ordinary each run encodes one current A map, Graph captures fixed bindings then replays fresh contents. Fresh standalone gate and original failure archives remain distinct.
- **Stall re-assessment:** Iter38–40 produced no observed >=3% own complete-call improvement; current39/44 SW CV in40 was0.0147–0.0907% with44 below0.063%, without a measured physical-floor claim. No repeat ncu after prior IPC failure. NVIDIA Hopper Tuning Guide describes one-thread TMA bulk transfer reducing SM transfer instructions/register payload; selective rawA4 differs from historical all-eight TMA31. Published Strassen-like A100/H200 results do not prove this H100 acceptance gate. Sources: https://docs.nvidia.com/cuda/hopper-tuning-guide/ ; https://www.microsoft.com/en-us/research/publication/compiling-strassen-like-matrix-multiplication-algorithms-to-fast-cuda-kernels/ ; https://github.com/microsoft/subcuber/blob/main/README.md .
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 5.856000 | 6.475200 | 0.904374 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 5.771200 | 6.630400 | 0.870415 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 5.876800 | 8.049600 | 0.730074 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 6.244800 | 8.104000 | 0.770582 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 6.388800 | 11.526400 | 0.554275 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 6.544000 | 11.472000 | 0.570432 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 13.513599 | 18.142401 | 0.744863 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 10.387200 | 18.136001 | 0.572739 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 32.120001 | 135.726404 | 0.236653 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 33.216000 | 135.292804 | 0.245512 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 87.284797 | 445.889616 | 0.195754 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 89.535999 | 438.430405 | 0.204219 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 195.870399 | 1084.815979 | 0.180556 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 205.849600 | 1092.535973 | 0.188414 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 11.646400 | 18.070400 | 0.644501 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 11.921600 | 18.051200 | 0.660433 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 8.470400 | 17.830400 | 0.475054 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_selective_tma_a_64x32 v45 | launch | 8.819200 | 17.881601 | 0.493200 | FAIL |

- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.
- **Next:** Mandatory samebinaryfull20 for45/39/44/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.
- **Archive:** trajectory/20261014_071808_iter-41/

#### Iter41 mandatory full1024 closure

Full samebinary20 rows45/39/44/36/Lt1,20warm100calls5groups3fresh32Lt8GiB. Original60/60 fresh numericalFAIL/NF0 retained; strict30=18/12NF0, SAN3zero, structural tests pass. New45 own latency decreases23.6931/23.2220% ordinary and26.2506/24.2674% Graph against39; ordinary Dense/SW0.688287/0.556617 remains below1 and precision gate fails. Full primary report records all groups/CSVauthority/event/batchwall scope; no promotion/AutoDense/goalactive.

### Iter42 — SW128 rawA SIMT control with unused currentA encode

- **Hypothesis:** Isolate rawA delivery with parent45 SW128 geometry/math/domain/host currentA map encode retained. New46 rawA vector8 SIMT and B SIMT, all128 writers each proxyfence before fullcommit; no kernel TMA issue/rawTX wait. The encoded map is deliberately unused, not a no-encode route.
- **Changes:** ONLY46 from frozen CPU candidate, BM64BN32BK64/2banks/wholeP3+4/ascending4K16/onecast/fixedU/RNalphaFMA/AutoDense and original floors preserved. Fresh compiled gate before first kernel, prior CAN45 not inherited.
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 128,128,128 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 5.844800 | 7.120000 | 0.820899 | FAIL |
| 128,128,128 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 5.985600 | 7.334400 | 0.816099 | FAIL |
| 256,256,256 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 6.056000 | 9.528000 | 0.635600 | FAIL |
| 256,256,256 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 6.108800 | 9.569600 | 0.638355 | FAIL |
| 512,512,512 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 6.356800 | 14.236800 | 0.446505 | FAIL |
| 512,512,512 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 6.475200 | 14.198400 | 0.456051 | FAIL |
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 13.185599 | 23.908800 | 0.551496 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 10.395200 | 23.659199 | 0.439372 | FAIL |
| 2048,2048,2048 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 31.996799 | 189.886403 | 0.168505 | FAIL |
| 2048,2048,2048 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 33.273599 | 189.639997 | 0.175457 | FAIL |
| 3072,3072,3072 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 86.854398 | 687.807989 | 0.126277 | FAIL |
| 3072,3072,3072 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 89.547199 | 685.172796 | 0.130693 | FAIL |
| 4096,4096,4096 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 196.054399 | 1815.388870 | 0.107996 | FAIL |
| 4096,4096,4096 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 206.263995 | 1804.337692 | 0.114316 | FAIL |
| 1024,512,1024 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 11.721600 | 23.763201 | 0.493267 | FAIL |
| 1024,512,1024 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 11.876800 | 23.644800 | 0.502301 | FAIL |
| 512,1024,1024 | bf16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 8.500800 | 23.665600 | 0.359205 | FAIL |
| 512,1024,1024 | fp16 | megakernel_wgmma_psplit_sw128_simt_control_64x32 v46 | launch | 8.828800 | 23.528001 | 0.375247 | FAIL |

- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.
- **Next:** Mandatory samebinaryfull20 for46/45/39/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.
- **Archive:** trajectory/20261014_074453_iter-42/


#### Iter42 mandatory full1024 closure

Mandatory samebinary20 rows46/45/39/36/Lt1,20warm100calls5groups3fresh32Lt8GiB completed with all five groups/authorityCSV retained. Original60/60 fresh numericalFAIL/NF0 and ten innerexit2 remain. SW128 SIMT46 with deliberately unused currentA map encode is slower than selectiveTMA45 by ordinary BF33.0511%/FP31.6073%, Graph35.6808%/31.7414%;45 remains fastest measured native. Current45 ordinary Dense/SW0.696120/0.558143 and Graph0.668873/0.553662 remain below1, original precision still fails. Strict30=18PASS12FAIL/NF0, SAN3 actualexit0 with zero summaries, structural default/target PASS; these do not certify FP64 floors. Full1830 code domain unexecuted. CPU-only coverage/metadata adapter failures and exact narrow corrections are archived without changing source/bin/raw or repeating GPU. Full primary report binds event/100-call batchwall scope, no isolated physical-cause or residency claim. AutoDense/goalactive; no automatic next iteration.

### Iter43 — BN64 selective rawA TMA with whole-stage RS operands

- **Hypothesis:** Increase BN32 toBN64 while retaining four rawA SW128 TMA boxes, SIMT finalB7, currentA map encode inside each ordinary call. The doubled N tile changes B/P/tail perCTA and grid shape; no isolated causality or residency claim.
- **Changes:** ONLY47, BM64BN64BK64/2banks/wholeP3+4/fourascendingK16/finalonecast/fixedU/RNalphaFMA/AutoDense and original floors retained. Initial40/224/240 actual STACK128spill168; fresh stage-local address reconstruction reduced STACK/spill to8, then48/216/240 same64512 pool eliminated spill. Every failed source/build/SASS is archived; final new compiled gate required before first47 kernel.
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_tma_bn64_64x64 v47 | launch | 14.793600 | 26.359999 | 0.561214 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_tma_bn64_64x64 v47 | launch | 10.571200 | 26.475200 | 0.399287 | FAIL |

- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.
- **Next:** Mandatory samebinaryfull20 for47/45/39/36/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.
- **Archive:** trajectory/20261014_081904_iter-43/


#### Iter43 mandatory full1024 closure

Mandatory samebinary20 rows47/45/39/36/Lt1,20warm100calls5groups3fresh32Lt8GiB completed. Own47 ordinaryBF25.9145593643/FP26.0444808006us vs45 17.6502394676/17.8268802166us: slower46.8227/46.0967%; Graph slower50.4893/48.4540%.45 remains fastest actually measured native. All five groups/200TIME/60CHECK originalFAIL/NF0 and10innerexit2 retained. DenseCV48.44% in47ordinaryBF and49.76% in39GraphFP are preserved, code ranking uses SW latency rather than Dense denominator. Strict30=18PASS12FAIL/NF0, SAN3 terminalexit0 with zero summaries, structuraldefault2862/2698 andselected60/140 pass; full1860 only unexecuted code domain. Fresh gate records actual initial-owned preINCcommonPzero acceptance truthfully; growth still follows successfulINC. All failed resource revisions archived without GPU launch. AutoDense/original precision goal remain active, no promotion/no physical cause/no ncu retry/no automatic next iteration.

### Iter44 — BN16 selective rawA TMA with whole-stage RS operands

- **Hypothesis:** Decrease BN32 toBN16 while retaining four rawA SW128 TMA boxes, SIMT finalB7, currentA map encode inside each ordinary call. The smaller N tile changes B/P/tail perCTA and grid shape; no isolated causality or residency claim.
- **Changes:** ONLY48, BM64BN16BK64/2banks/wholeP3+4/fourascendingK16/finalonecast/fixedU/RNalphaFMA/AutoDense and original floors retained. Initial40/128/128 actual REG128 STACK168spill224; stage-local volatile thread-id reconstructs the same rawA coordinates, with independent B onecast scheduling preserving original unrounded FP32 DAG. Revision1 retains40/128/128 and eliminates STACK/LOCAL/spill. Fresh I128 owned49152 exceeds requested37888; no two-CTA residency claim. Every failed source/build/SASS is archived; final new compiled gate required before first48 kernel.
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; failed numerical results retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_tma_bn16_64x16 v48 | launch | 13.262400 | 27.675200 | 0.479216 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_tma_bn16_64x16 v48 | launch | 10.424000 | 28.675199 | 0.363520 | FAIL |

- **Analysis:** Own complete SW latency ranks code changes; Dense timing variation and machine/fixture/SAN PASS do not certify original FP64 numerical accuracy.
- **Next:** Mandatory samebinaryfull20 for48/45/47/39/Lt1,100calls5groups3fresh irrespectivequick3%, originalstrict30 andserialSAN3, coverage/profile/report/twoindependentfinal audits.
- **Archive:** trajectory/20261014_084520_iter-44/


### Iteration44 closure — BN16 TMA negative result; original goal remains active

Mega48 is a new separately compiled BM64/BN16/BK64 selective rawA4 TMA route, with B SIMT, two banks,384 threads,whole-stage3+4 products/four ascendingK16,finalonecast,fixedU,RNalpha-betaFMA and zero global matrix workspace. Parent45 and all old implementations/evidence remain unchanged. Source candidate quota40/128/128/min1 initially compiled REG128/STACK168/spill224; all56 spilledwords were consumer pre-MMA address/CTA-base temporaries, producer had no locals. Revision1 stage-local rawA coordinates and independent finalB cast scheduling retained the original FP32 RN graph and eliminated STACK/LOCAL/spill. Fresh independent gate uses I128 owned49152≥request37888, role maxima37/125/124, actual dynamic95232/1024alignment, Driverstatic0/toolSHARED1024 separately, occupancy API predicted1—not measured residency.

Mandatory full20 SAMEBINARY routes48/45/47/39/Lt1:20warm100calls5groups3fresh/32Lt8GiB/rotate1, all original groups retained. New48 ordinary BF27.426879406µs/FP28.2748794556µs vs45 BF17.6649594307/FP17.799680233µs: own latency worsens55.26149%/58.85049%; Graph48 BF26.989440918/FP27.7884793282µs vs45 BF16.9500803947/FP17.3484802246µs worsens59.22898%/60.17818%. Four declared comparisons48vs45,48vs47,45vs39,48vs39 yield16 pairs, all noncausal.45 remains fastest measured native in each dtype/mode; FP16Graph Lt1 is slightly faster than45 overall, so native ranking is not all-route ranking. Dense maxCV0.91566%/SW maxCV0.12310%; no trimming or retry.

Original full20 numerical gate:60CHECK FAIL/nonfinite0, all10innerexit2. Newstrict30:18PASS12FAIL/nonfinite0,24native+6Densefallback and30separateDense references (CASE60); full1890 is code-domain only, unexecuted. Actualdefault2922CASE/2836PASS/5284BEGIN exit0, selected48 60CASE/140PASS/204BEGIN exit0; rawU84 CROSSATOM_N16_N32 strict-bit fixtures use native45 at shared parentN64, total246 bound runs=84reference+162candidate, separate minimumN32 native basic/Graph fixtures. NewGraph40configs/120freshreplays; SANmem/race/sync eachactual0 with zero errors/hazards/warnings; onlymem measures leak0. HugeA-stride metadata10 helper guards pass but all10Plan setups unavailable (cuBLASstatus15), allocated_input_bytes/encodes/native_runs0: no huge GPU fallback execution certificate. Structural/SAN PASS does not certify original FP64 precision. AutoDense, original floors and goal remain unchanged/active.

Timing/source provenance: quicksourceGit987ef70 preceded final plan creation because original generator rejected actual48 field names. Original scripts andCPUexit1 diagnostic remain archived; new distinct ABI readers preserve original gate383ca8b8. Plan19998bada and once-complete identity5454545e were exclusively created before full timing; sourceGit proves seven executed sources, laterprestatus Git proves plan/identity. Postrun clarification67844069 is explicit, never claimed a preexecution artifact; identity/plan were never appended/reconstructed. Original test prepared7d957 generator first actualCPUexit0 with no adapters.

Reassessment after42–44: none gives≥3% own-latency improvement over best45; current44 SW timing is stable within the retained group statistics. No measured physical-floor/stall/traffic or occupancy-cause conclusion; no Ncu retry. Next work should reconsider register residency/lifetimes, rather than assume smallerBN alone helps. NVIDIA Hopper CUDA12.4.1 tuning documents64Kregisters/SM and PTX setmaxnreg documents CTA-internal redistribution; pool arithmetic alone cannot establish twoCTA residency. Primary sources: https://docs.nvidia.com/cuda/archive/12.4.1/hopper-tuning-guide/index.html#occupancy and https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg . Frozen private min2 future44 q40/88/112 is CPU-only/uncompiled; actual48 role125/124 flags pressure risk without proving future compilation outcome. No49/public diagnostic/nextiteration automatically started.

Evidence: results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_REPORT.md; small_psplit_tma_bn16_measurements_iter44.json/profile_iter44.json; psplit_tma_bn16_test_coverage_iter44.json; immutable compiled gate383ca8b8 and failedinitialf9f3d83a.

### Iter45 — BN16 minCTA2 with atom-local RS inputs

- **Hypothesis:** Test register residency and shorter transformed-A lifetime on BN16/BK64 while retaining four rawA TMA boxes and SIMT finalB7. MinCTA2 is a source constraint; occupancy API predicts2, measured residency is unknown.
- **Changes:** ONLY49 q40/88/112, BM64BN16BK64/2banks/384threads. Known48 stage-local raw coordinates and B finalcast scheduling preserve RN. Firstcompile actual I80 STACK24/spill40FP16/36BF16 and newC7512 prohibited launch; failed source/PTX/SASS/resources retained. Revision1 uses one K16 A atom per P, four ascending atom windows each build/fence/3or4HGMMA/commitwait0/postfences, release after all4; whole FP32P/wholeK/fixedU/RNalpha-betaFMA/original floors unchanged. Fresh roles37/77/71, zeroSTACKLOCALspill, owned30720=request30720. Legal initial-owned commonP R32..55 preINC zeros recorded truthfully; no meaningful growth access. Fresh independent CAN49 only.
- **Bench:** quick20warm/20calls/5groups/1fresh, screen only; all originalgate failures retained.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_tma_bn16_min2_64x16 v49 | launch | 13.097601 | 26.012799 | 0.503506 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_tma_bn16_min2_64x16 v49 | launch | 10.420800 | 25.880000 | 0.402658 | FAIL |

- **Analysis:** Complete own SW latency evaluates the combined resource/lifetime change. No measured residency/stall/traffic cause or numerical certification is inferred. Current A encode remains inside each ordinary complete call.
- **Next:** Mandatory samebinaryfull20 for49/48/45/39/Lt1,20warm100calls5groups3fresh regardlessquick3%; final plan and once-complete identity created after this quick Git checkpoint but before full timing. SourceGit proves seven executed sources; later unique prestatus Git binds plan/identity. Originalstrict30 andserialSAN3, coverage/profile/report and independentfinal audits follow.
- **Archive:** trajectory/20261014_091648_iter-45/

## Iteration45 actual closed evidence

- ONLY49 BM64 BN16 BK64/minCTA2/q40-88-112 uses four ascending K16 atom-local A windows while wholeK FP32 P, fixed U, original RN DAG/final onecasts and alphaRN/betaFMA remain unchanged. Initial whole-stage49 REG80 STACK24/spill40FP16/36BF16 and newC7512 forbade any native launch; revision1 atom-local A repaired those resources. Fresh actual I80 owns30720=request30720, all-role implicit maxima37/77/71 and zero STACK/LOCAL/spill; legal initial-owned commonP R32..55 zeros precede INC, no actual grown registers used. Dynamic95232 and occupancy API prediction2 are recorded; measured residency remains null.
- Samebinary FULL20 declares49/48/45/39/Lt1 × two dtype × ordinary/Graph,20warm100calls5groups3fresh/32Lt8GiB/rotate1; all five groups and original60FAIL/NF0 retained. Authority CSV ranks own complete SW latency, not changes in independent Dense denominator.
- bf16/launch:49 25.678400993us vs48 27.596158981us vs45 17.748160362us.49/48 latency reduction 6.949366%;49/45 increase 44.682043%.45 is the fastest measured native route, not necessarily fastest among all routes.
- bf16/graph:49 25.517759323us vs48 27.116799355us vs45 17.063360214us.49/48 latency reduction 5.896861%;49/45 increase 49.547094%.45 is the fastest measured native route, not necessarily fastest among all routes.
- fp16/launch:49 25.307199955us vs48 28.145918846us vs45 17.865279913us.49/48 latency reduction 10.085721%;49/45 increase 41.655771%.45 is the fastest measured native route, not necessarily fastest among all routes.
- fp16/graph:49 24.939839840us vs48 27.740480900us vs45 17.331520319us.49/48 latency reduction 10.095863%;49/45 increase 43.898743%.45 is the fastest measured native route, not necessarily fastest among all routes.
- Actual default CASE/PASS/BEGIN 2982/2974/5488 and selected49 60/140/204 each exit0. Strict original30 gates 18PASS/12FAIL/nonfinite0; three serial CUDA12.8 sanitizers exit0 with zero reported errors/hazards/leaks within their tested scopes. Full1920 numerical source domain was not executed; structural/cross-atom bit fixtures do not certify original FP64 precision.
- Plan/onceidentity were created after quick source Git65a2292 and before full timing. SourceGit binds seven executed sources; later prestatusGit binds the SHA-pinned plan/identity. No hindsight identity append or timing rerun. Auto remains Dense, precision_certified=false and goal_complete=false; no automatic iteration46/public50. Full report: results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_MIN2_REPORT.md.
- Separate ROOT CPU-only precision future:30 frozen files,27cases/111schemes; mixed BF16-to-FP16 reduced error on7/7 random cases but bounded exact finite cancellation still fails BF16 gate≈0.001 with mixed eF9.7169 and hi+lo15 omitting lo*lo eF0.06917. Large/tiny stress includes nonfinite/eF1. Original failed CPU helper exit1 retained. No CUDA/GPU/performance/precision-policy change or certification follows from that proxy evidence; results/psplit_tma_bn16_min2_precision_future_evidence_iter45.json.

### Iter46 — paired K16 inputs, two K32 windows on BN16 TMA

- **Hypothesis:** Retain two successive K16 A atoms until one matching wait0, reducing four input windows in49 to two in50. Geometry64x16x64,2banks/384threads,40/96/96/minCTA2. More live A may increase register pressure; no speed or measured residency is inferred from the source constraint.
- **Changes:** ONLY50 new TU. Each product retains ascending atom0..3, original FP32RN operand DAG and one final Element cast, wholeK seven FP32P, fixedU and RNalpha/betaFMA. Original guards/floors/AutoDense/37earlyblock/zero globalmatrixworkspace unchanged. Fresh two-dtype machine/ABI/link/init/pending review and occupancy attributes are bound independently before launch; original instance fields remain authoritative.
- **Bench:** AKO quick20warm/20calls/5groups/1fresh. All original numerical FAILs and exits retained; full timing follows regardlessquick3%.

| M,N,K | dtype | route | mode | Dense us | complete SW us | Dense/SW | originalgate |
|---|---|---|---|---:|---:|---:|---|
| 1024,1024,1024 | bf16 | megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 v50 | launch | 13.094400 | 26.059201 | 0.502487 | FAIL |
| 1024,1024,1024 | fp16 | megakernel_wgmma_psplit_tma_bn16_k32_window_64x16 v50 | launch | 10.384000 | 25.609601 | 0.405473 | FAIL |

- **Analysis:** Whole-call own latency evaluates the input-window schedule. Static occupancy API is a prediction, not measured residency or causal stall proof; numerical certification remains the original independent gate. Ordinary50 includes current-A encode.
- **Next:** After this quick Git checkpoint, freeze mandatorysamebinary20 (50/49/45/39/Lt1) and separate supplemental8 originalH10 single-P input fusion/two launches andH12 S/T once/three launches before anyfull timing. All28 use20warm100calls5groups3fresh/rotate1/32Lt8GiB. Once-complete identity first-creation includes bothplanSHA; sourceGit proves7 executed sources, laterprestatus Git binds plans/identity. Strict30/serialSAN3/coverage/independent audits follow; no automatic47.
- **Archive:** trajectory/20261014_094446_iter-46/

## Iteration46 actual closed evidence

- ONLY50 BM64BN16BK64/minCTA2/q40-96-96 retains two successive K16 A atoms per input window, two windows per stage. Per-P ascendingK16/wholeK sevenP/originalFP32RN operand DAG/finalonecast/fixedU/alphaRNbetaFMA/floors unchanged. ActualI80 owns30720 >=requested29696; fresh full-role implicit maxima37/91/88; STACK/LOCAL/spill0/no new50C7512. PhysicalA24/32+wholeP24/32 have6/8 HGMMA per matchingwait0, postwait fences before reuse. Legal initial-owned commonP preINC zero is recordedfalse-afterINC/true-owned; meaningful grownR80..91 and80..88 accesses followsuccessfulINC and definite definitions. Dyn95232/Driverstatic0/toolSHARED1024/APIpred2 distinct; measuredresidency remainsnull.
- Samebinarymandatory20 (50/49/45/39/Lt1) plus separately predeclared supplemental8 originalH10 fused input/two kernels andH12 ST-only-once/three kernels: all28 use20warm100calls5groups3fresh/32Lt8GiB/rotate1, GPUevent completecall and100call synchronizedbatchwall/call. Main200TIME/60CHECK plus supplemental80TIME/24CHECK, all84originalFAIL/NF0 and14innerexit2 preserved. All5groups retained.
- launch/bf16:50 25.792961121us vs49 25.807039738us vs45 17.776319981us.50/49 own latency change -0.054553%: no>=3% gain. H10 38.329598904us, H12 34.260160923us: ST-once route own decrease 10.616959%.45 is fastest measured native inallfour currentrows; it isalso fastest ofallroutes inthisbatchonly.
- launch/fp16:50 25.290238857us vs49 25.263359547us vs45 17.894400358us.50/49 own latency change +0.106396%: no>=3% gain. H10 38.427839279us, H12 34.536321163us: ST-once route own decrease 10.126820%.45 is fastest measured native inallfour currentrows; it isalso fastest ofallroutes inthisbatchonly.
- graph/bf16:50 25.419840813us vs49 25.414400101us vs45 16.967359781us.50/49 own latency change +0.021408%: no>=3% gain. H10 36.274559498us, H12 31.418879032us: ST-once route own decrease 13.385912%.45 is fastest measured native inallfour currentrows; it isalso fastest ofallroutes inthisbatchonly.
- graph/fp16:50 24.968318939us vs49 24.995200634us vs45 17.350720167us.50/49 own latency change -0.107547%: no>=3% gain. H10 36.746239662us, H12 31.959040165us: ST-once route own decrease 13.027726%.45 is fastest measured native inallfour currentrows; it isalso fastest ofallroutes inthisbatchonly.
- Actualdefault CASE/PASS/BEGIN 3042/3112/5692, selected50 60/140/204 exits0. Strictoriginal30 18PASS/12FAIL/NF0;30separateDense references give60CASE, full1950codeonlynotexecuted. SAN3actualexit0/errors0/memleak0/racehazardswarnings0; nonmem leaksnull. CROSSATOMref45 rawU84ref+162candidate=246total andGraph40/120fresh are structural scopes, not generalFP64 certification.
- Quick AKOexit2 thennexttwoowner toolcallslogger/Git9988cce. Main7c10de36/supp0e5f1eb3 andonceidentitya4afcdd2 werefirst-createdafterquickGit/beforeANYfulltiming; sourceGit binds7sources only, laterprestatusGit bindsplans/identity. Prepared9aa CPUfirstexit1 was only inherited SAN alias comparison, preserved; distinctsingle-token adapter is explicitlypostvalidationfalse, originalpreparedtrue/actualfrozencommands preserved. No GPU rerun. Huge-stride10setup_unavailable/status15/zeroallocation-encode-native, no successfulPlanrouteclaims. AutoDense/precisionfalse/goalfalse; no automaticiteration47/public51.
- Separate next-designCPU geometry package7files: inplaceSS BM64BN32BK32/q40-88-104/I80 is uncompiledhypothesis; BN16 sibling I64 alsounproven.128CTAs for132SM does not prove two-CTAbenefit. No sourceintegration/CuTe/compiler/GPU/numerical/performance proof; geometry actualCPUexit0 only. OriginaloldCPUprecisionfuture remainshistorical andnot certified. Full report: results/1024_MEGAKERNEL_PSPLIT_TMA_BN16_K32_WINDOW_REPORT.md.

### Iter47 — cooperative in-place shared ST, SS WGMMA BN32/BN16

- **Hypothesis:** Eight raw quadrant TMA boxes load into the first four shared final planes, then all256 consumers own disjoint packed pairs and construct final S/T once per CTA/bankstage. This is not global H12 ST-once. Both current A/B maps encode inside each ordinary call. More consumer transform work/bar1 may offset avoiding redundant constructions; no source-only speed claim.
- **Changes:** ONLY51/52 new shared TU with original RN DAG/final onecast/wholeK sevenP/fixedU/alphaRNbetaFMA unchanged. Initial frontend failures, CPU byte-flag consumer diagnostics and BN32 min2 complete-P spills retained. Final51 min1 q40/128/128,52 min2 q48/80/80; fresh four-instance independent compiled gate binds final source/bin streams, role initialization/actual pool/pending/layout/TMA/sync. Initial-owned preINC zeros remain truthful.
- **Bench:** One combined serial quick51+52,20warm20calls5groups1fresh. Original gates/exit2 retained; full24 timing required regardless3% gain.

| dtype | native route | Dense us | SW complete us | Dense/SW | original gate |
|---|---|---:|---:|---:|---|
| bf16 | megakernel_wgmma_psplit_tma_inplace_ss_64x32 v51 | 13.249600 | 25.084800 | 0.528192 | FAIL |
| fp16 | megakernel_wgmma_psplit_tma_inplace_ss_64x32 v51 | 10.488000 | 25.331199 | 0.414035 | FAIL |
| bf16 | megakernel_wgmma_psplit_tma_inplace_ss_64x16 v52 | 13.313600 | 31.312001 | 0.425192 | FAIL |
| fp16 | megakernel_wgmma_psplit_tma_inplace_ss_64x16 v52 | 10.416000 | 31.011200 | 0.335879 | FAIL |

- **Analysis:** Own SW complete-call latency will rank routes, not Dense denominator changes. Static occupancy API is not measured residency or causality; no precision promotion.
- **Next:** After this quick source Git checkpoint, create immutable full24 plan and once-complete identity before first full timing:51/52/45/50/39/Lt1 ×two dtype ×ordinary/Graph;20warm100calls5groups3fresh32Lt8GiBrotate1. Six axes×four modes=24, strict30each with shared30Dense refs, SAN3each serial. Fresh H10/H12 remain iteration46 history. Close unique prestatus and independent audits; no automatic48.
- **Archive:** trajectory/20261014_104756_iter-47/
