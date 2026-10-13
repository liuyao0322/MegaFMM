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
