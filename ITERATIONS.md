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

- Resource probe: megakernel64x32 M1=150 registers,42KiB SMEM,0 spill;64x64 M1=254 registers,56KiB,0 spill. Stock Hopper variants=168 static registers with512/1024-byte compiler spill stores/loads.
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
