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
